"""Subconscious Controller — PyTorch Model, ONNX Export, and Retraining.

Defines :class:`SubconsciousController`, a small MLP (~50 K parameters)
that maps a 32-dimensional system-state vector to an 8-dimensional
control-signal vector.  The model is trained in PyTorch, then exported
to ONNX (with optional quantization) for deployment on the NPU,
DirectML, or CPU backend.

Also provides :func:`retrain_controller` for retraining the controller
from accumulated telemetry data (Year 2 Q2, Workstream F.3).

Architecture::

    Linear(32 → 128) → ReLU → Dropout(0.1)
    Linear(128 → 128) → ReLU → Dropout(0.1)
    Linear(128 → 8)   → output head: Sigmoid/Softplus/Tanh per slice

The output head applies per-channel activation functions to enforce
physical constraints on the control signals (see :meth:`forward`).

Example:
    >>> import torch
    >>> model = SubconsciousController()
    >>> z = torch.randn(4, 32)
    >>> ctrl = model(z)
    >>> print(ctrl.shape)
    torch.Size([4, 8])
    >>> model.export_to_onnx("subconscious_controller.onnx")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prinet.core.subconscious import CONTROL_DIM, STATE_DIM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIDDEN: int = 128
"""Default hidden-layer width."""

_DROPOUT: float = 0.1
"""Dropout probability for regularization."""


# ---------------------------------------------------------------------------
# SubconsciousController
# ---------------------------------------------------------------------------


class SubconsciousController(nn.Module):
    """Small MLP that maps system state → control signals.

    The architecture is deliberately small (~50 K parameters) so it can
    run on the NPU or CPU with sub-millisecond latency.

    Args:
        state_dim: Dimensionality of the input state vector.
        hidden: Width of the hidden layers.
        control_dim: Dimensionality of the output control vector.
        dropout: Dropout probability (0 disables).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden: int = _HIDDEN,
        control_dim: int = CONTROL_DIM,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, control_dim),
        )

        # Initialize weights with small fan-in-scaled values
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming uniform and zero biases."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z: Tensor) -> Tensor:
        """Compute control signals from system state.

        Output layout (8 dims):
            +---------+---------------------------------------------------------+
            | Indices | Meaning & activation                                    |
            +=========+=========================================================+
            | 0–1     | ``suggested_K_min, K_max`` — Softplus (positive)        |
            | 2       | ``lr_multiplier`` — Softplus (positive)                 |
            | 3–5     | ``regime weights`` — Softmax (probability simplex)      |
            | 6       | ``alert_level`` — Sigmoid ``[0, 1]``                    |
            | 7       | ``coupling_mode_suggestion`` — raw logit                |
            +---------+---------------------------------------------------------+

        Args:
            z: State tensor of shape ``(B, state_dim)``.

        Returns:
            Control tensor of shape ``(B, control_dim)``.
        """
        raw = self.net(z)  # (B, 8)

        # Apply per-channel activations to enforce physical bounds
        k_range = F.softplus(raw[:, 0:2])  # positive K bounds
        lr_mult = F.softplus(raw[:, 2:3])  # positive multiplier
        regime = F.softmax(raw[:, 3:6], dim=-1)  # probability simplex
        alert = torch.sigmoid(raw[:, 6:7])  # [0, 1]
        coupling = raw[:, 7:8]  # raw logit (floored later)

        return torch.cat([k_range, lr_mult, regime, alert, coupling], dim=-1)

    # ------------------------------------------------------------------
    # ONNX Export
    # ------------------------------------------------------------------

    def export_to_onnx(
        self,
        output_path: str | Path,
        *,
        opset_version: int = 18,
        dynamic_batch: bool = True,
    ) -> Path:
        """Export the model to ONNX format.

        The exported graph uses the standard input name
        ``"state_vector"`` and output name ``"control_signals"``
        expected by :func:`~prinet.utils.npu_backend.create_session`.

        Args:
            output_path: Destination ``.onnx`` file.
            opset_version: ONNX opset to target.
            dynamic_batch: If ``True``, mark the batch dimension as
                dynamic so the model accepts any batch size.

        Returns:
            Resolved :class:`Path` to the written ONNX file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()
        dummy = (torch.randn(1, self.state_dim),)

        dynamic_axes = (
            {"state_vector": {0: "batch"}, "control_signals": {0: "batch"}}
            if dynamic_batch
            else None
        )

        torch.onnx.export(
            self,
            dummy,
            str(output_path),
            input_names=["state_vector"],
            output_names=["control_signals"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

        logger.info("ONNX model exported to %s", output_path)
        return output_path.resolve()

    # ------------------------------------------------------------------
    # ONNX Quantization
    # ------------------------------------------------------------------

    @staticmethod
    def quantize_onnx(
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        quant_format: str = "QDQ",
    ) -> Path:
        """Quantize an ONNX model to INT8.

        Uses :mod:`onnxruntime.quantization` dynamic quantization.
        ``QDQ`` format is preferred for NPU / DirectML deployment.

        Args:
            input_path: Source ``.onnx`` file (float32).
            output_path: Destination ``.onnx`` file.  If ``None``,
                appends ``_int8`` to the input stem.
            quant_format: Quantization format (``"QDQ"`` or ``"QOperator"``).

        Returns:
            Resolved :class:`Path` to the quantized ONNX file.

        Raises:
            ImportError: If ``onnxruntime.quantization`` is unavailable.
        """
        try:
            from onnxruntime.quantization import (
                QuantFormat,
                QuantType,
                quantize_dynamic,
            )
        except ImportError as exc:
            msg = (
                "onnxruntime.quantization is required for INT8 conversion.  "
                "Install it with:  pip install onnxruntime"
            )
            raise ImportError(msg) from exc

        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_stem(input_path.stem + "_int8")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fmt = QuantFormat.QDQ if quant_format == "QDQ" else QuantFormat.QOperator

        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            extra_options={"MatMulConstBOnly": True},
        )

        logger.info("Quantized ONNX model written to %s", output_path)
        return output_path.resolve()

    # ------------------------------------------------------------------
    # Convenience: parameter count
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters.

        Returns:
            Integer count of all parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================================================================
# Year 2 Q2 — Workstream F.3: Controller Retraining Pipeline
# =========================================================================


def retrain_controller(
    telemetry_path: str | Path | None = None,
    telemetry_records: list[dict[str, Any]] | None = None,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    output_onnx_path: str | Path = "models/subconscious_controller.onnx",
    seed: int = 42,
) -> tuple[SubconsciousController, dict[str, Any]]:
    """Retrain SubconsciousController from accumulated telemetry data.

    Loads telemetry (state → control pairs), trains a fresh
    :class:`SubconsciousController`, and exports the result to ONNX.

    The training target is the control signals that were recorded during
    observation mode. This implements supervised learning:
    state_vector → optimal_control_signals.

    If the telemetry doesn't contain control signals, the function
    will generate synthetic targets using the mean-field heuristic.

    Args:
        telemetry_path: Path to telemetry JSON file (from
            :meth:`TelemetryLogger.to_json`). Provide either this
            or ``telemetry_records``.
        telemetry_records: List of telemetry record dicts (from
            :attr:`TelemetryLogger.records`). Overrides ``telemetry_path``.
        n_epochs: Training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        output_onnx_path: Path for the exported ONNX model.
        seed: Random seed.

    Returns:
        ``(controller, metrics)`` where controller is the retrained model
        and metrics is a dict with ``train_loss``, ``n_samples``, etc.
        Values may be ``float``, ``int``, or ``str``.

    Raises:
        ValueError: If no telemetry source is provided or data is empty.

    Example:
        >>> ctrl, metrics = retrain_controller(
        ...     telemetry_path="telemetry.json",
        ...     n_epochs=20,
        ... )
        >>> print(f"Final loss: {metrics['train_loss']:.4f}")
    """
    import json as _json

    from prinet.core.subconscious import CONTROL_DIM, STATE_DIM

    torch.manual_seed(seed)

    # Load telemetry data
    if telemetry_records is not None:
        records = telemetry_records
    elif telemetry_path is not None:
        with open(telemetry_path, "r") as f:
            records = _json.load(f)
    else:
        raise ValueError("Provide either telemetry_path or telemetry_records.")

    if len(records) == 0:
        raise ValueError("Telemetry data is empty.")

    # Build state and control tensors from telemetry
    states: list[Tensor] = []
    controls: list[Tensor] = []

    for rec in records:
        # Construct state vector (32 dims)
        rpb = rec.get("r_per_band", [0.5, 0.5, 0.5])
        rg = rec.get("r_global", sum(rpb) / len(rpb))
        loss_val = rec.get("loss", 0.0)
        epoch = rec.get("epoch", 0)

        # State: [r_per_band(3), r_global(1), loss(1), epoch_norm(1),
        #         zeros(26)] → 32 dims
        state_vec = torch.zeros(STATE_DIM)
        state_vec[0:3] = torch.tensor(rpb[:3], dtype=torch.float32)
        state_vec[3] = float(rg)
        state_vec[4] = float(loss_val)
        state_vec[5] = float(epoch) / 100.0  # Normalized epoch
        states.append(state_vec)

        # Control target (8 dims)
        ctrl_data = rec.get("control", None)
        if ctrl_data is not None and isinstance(ctrl_data, dict):
            ctrl_vec = torch.tensor(
                [
                    ctrl_data.get("suggested_K_min", 0.5),
                    ctrl_data.get("suggested_K_max", 5.0),
                    ctrl_data.get("lr_multiplier", 1.0),
                    ctrl_data.get("regime_mf_weight", 0.5),
                    ctrl_data.get("regime_sk_weight", 0.3),
                    ctrl_data.get("regime_full_weight", 0.2),
                    ctrl_data.get("alert_level", 0.0),
                    ctrl_data.get("coupling_mode_suggestion", 0.0),
                ],
                dtype=torch.float32,
            )
        else:
            # Synthetic target: mean-field heuristic
            ctrl_vec = torch.tensor(
                [
                    0.5,
                    5.0,
                    1.0,  # K_min, K_max, lr_mult
                    0.5,
                    0.3,
                    0.2,  # regime weights
                    max(0.0, min(1.0, float(loss_val))),  # alert
                    0.0,  # coupling mode
                ],
                dtype=torch.float32,
            )
        controls.append(ctrl_vec)

    state_tensor = torch.stack(states)  # (N, 32)
    control_tensor = torch.stack(controls)  # (N, 8)

    # Create and train controller
    controller = SubconsciousController(
        state_dim=STATE_DIM,
        control_dim=CONTROL_DIM,
    )
    controller.train()

    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(state_tensor, control_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    final_loss = 0.0
    for _epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for state_batch, ctrl_batch in loader:
            optimizer.zero_grad()
            pred = controller(state_batch)
            loss = F.mse_loss(pred, ctrl_batch)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)

    # Export to ONNX
    output_path = controller.export_to_onnx(output_onnx_path)

    metrics = {
        "train_loss": final_loss,
        "n_samples": len(records),
        "n_epochs": n_epochs,
        "onnx_path": str(output_path),
    }

    return controller, metrics
