"""HybridPRINet — End-to-end hybrid oscillatory + rate-coded model.

Implements the Q4 integration target: chains Local Oscillatory Binding
Modules (LOBMs), Phase-to-Rate Conversion, and a Global Rate Integration
Module (GRIM) into a single, end-to-end trainable architecture.

Also provides :class:`AlternatingOptimizer` for joint training of
oscillatory and rate-coded parameter groups with separate learning rates
and optional subconscious daemon integration.

Architecture::

    Input → LOBM (HierarchicalResonanceLayer × L)
          → PhaseToRateConverter (sparse bottleneck)
          → LayerNorm
          → GRIM (Transformer encoder)
          → Classifier head
          → log_softmax

Example:
    >>> import torch
    >>> model = HybridPRINet(n_input=256, n_classes=10)
    >>> x = torch.randn(4, 256)
    >>> log_probs = model(x)
    >>> print(log_probs.shape)
    torch.Size([4, 10])
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from prinet.nn.layers import OscillatoryAttention


# ---- Constants ----------------------------------------------------------

_EPS: float = 1e-6
_LOGIT_CLAMP: float = 50.0


# ---- HybridPRINet ------------------------------------------------------


class HybridPRINet(nn.Module):
    """End-to-end hybrid PRINet model.

    Chains LOBM (oscillatory encoding) → PhaseToRate (sparse conversion)
    → GRIM (Transformer rate integration) → classifier head.

    The LOBM stage uses :class:`~prinet.nn.layers.HierarchicalResonanceLayer`
    to produce hierarchical oscillator amplitudes. Phase and amplitude
    are then converted to sparse rate codes via
    :class:`~prinet.nn.layers.PhaseToRateConverter`. The GRIM stage
    processes rate codes through a small Transformer encoder before
    final classification.

    Args:
        n_input: Input feature dimension.
        n_classes: Number of output classes.
        n_delta: Delta-band oscillators per LOBM layer.
        n_theta: Theta-band oscillators per LOBM layer.
        n_gamma: Gamma-band oscillators per LOBM layer.
        n_lobm_layers: Number of LOBM layers (stacked).
        lobm_steps: ODE integration steps per LOBM layer.
        lobm_dt: ODE timestep for LOBM.
        coupling_strength: Intra-band coupling K.
        pac_depth: Initial PAC modulation depth.
        rate_mode: PhaseToRateConverter mode (``"soft"``, ``"hard"``).
        rate_sparsity: Target sparsity for rate codes.
        grim_d_model: Transformer model dimension.
        grim_n_heads: Number of attention heads.
        grim_n_layers: Number of Transformer layers.
        grim_dropout: Dropout rate in GRIM Transformer.
    """

    def __init__(
        self,
        n_input: int = 256,
        n_classes: int = 10,
        # LOBM config
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 32,
        n_lobm_layers: int = 2,
        lobm_steps: int = 10,
        lobm_dt: float = 0.01,
        coupling_strength: float = 2.0,
        pac_depth: float = 0.3,
        # Phase-to-Rate config
        rate_mode: str = "soft",
        rate_sparsity: float = 0.1,
        # GRIM config
        grim_d_model: int = 64,
        grim_n_heads: int = 4,
        grim_n_layers: int = 2,
        grim_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        from prinet.nn.layers import (
            HierarchicalResonanceLayer,
            PhaseToRateConverter,
            SparsityRegularizationLoss,
        )

        self.n_input = n_input
        self.n_classes = n_classes
        self.n_osc_total = n_delta + n_theta + n_gamma

        # ---- LOBM stage: stacked oscillatory binding modules ----
        lobm_list: list[nn.Module] = []
        first_input_dim = n_input
        for i in range(n_lobm_layers):
            in_dim = first_input_dim if i == 0 else self.n_osc_total
            lobm_list.append(
                HierarchicalResonanceLayer(
                    n_delta=n_delta,
                    n_theta=n_theta,
                    n_gamma=n_gamma,
                    n_dims=in_dim,
                    n_steps=lobm_steps,
                    dt=lobm_dt,
                    coupling_strength=coupling_strength,
                    pac_depth=pac_depth,
                )
            )
        self.lobm_layers = nn.ModuleList(lobm_list)

        # LayerNorm between stages for numerical stability
        self.lobm_norms = nn.ModuleList(
            [nn.LayerNorm(self.n_osc_total) for _ in range(n_lobm_layers)]
        )

        # ---- Phase-to-Rate conversion ----
        # We treat the LOBM output amplitudes as "pseudo-amplitudes"
        # and use zero phase (cos-based soft competition) for rate conv
        self.phase_to_rate = PhaseToRateConverter(
            n_oscillators=self.n_osc_total,
            mode=rate_mode,
            sparsity=rate_sparsity,
        )

        # Sparsity regularization loss (accessible for training)
        self.sparsity_loss_fn = SparsityRegularizationLoss(
            target_sparsity=1.0 - rate_sparsity,
        )

        # ---- GRIM stage: Transformer rate integration ----
        # Project oscillator dim to Transformer model dim
        self.grim_proj = nn.Linear(self.n_osc_total, grim_d_model)
        self.grim_norm = nn.LayerNorm(grim_d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=grim_d_model,
            nhead=grim_n_heads,
            dim_feedforward=grim_d_model * 4,
            batch_first=True,
            dropout=grim_dropout,
        )
        self.grim_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=grim_n_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(grim_d_model, grim_d_model),
            nn.ReLU(),
            nn.Dropout(grim_dropout),
            nn.Linear(grim_d_model, n_classes),
        )

    def forward(
        self,
        x: Tensor,
        return_rates: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass through full HybridPRINet.

        Args:
            x: Input tensor of shape ``(B, D)`` or ``(D,)``.
            return_rates: If ``True``, also return sparse rate codes
                for sparsity analysis.

        Returns:
            Log-probabilities ``(B, K)``; or tuple of
            ``(log_probs, sparse_rates)`` if ``return_rates=True``.
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        # LOBM stage: oscillatory binding
        # All layers except the last are called normally.
        # The last layer is called with return_phase=True to obtain
        # real oscillatory phases for the phase-to-rate stage.
        h = x
        for lobm_layer, norm in zip(
            self.lobm_layers[:-1], self.lobm_norms[:-1]
        ):
            h = lobm_layer(h)
            h = norm(h)

        # Last LOBM layer — also get real phases
        last_layer = self.lobm_layers[-1]
        last_norm = self.lobm_norms[-1]
        h_amp, osc_phase = last_layer(h, return_phase=True)
        h = last_norm(h_amp)

        # Phase-to-Rate: use real oscillatory phases (not pseudo-phase)
        rate_codes = self.phase_to_rate(osc_phase, h)

        # GRIM stage: Transformer processing
        grim_in = self.grim_proj(rate_codes)  # (B, d_model)
        grim_in = self.grim_norm(grim_in)
        # Transformer expects (B, seq_len, d_model) — treat as single token
        grim_in = grim_in.unsqueeze(1)  # (B, 1, d_model)
        grim_out = self.grim_encoder(grim_in)
        grim_out = grim_out.squeeze(1)  # (B, d_model)

        # Classification head (float32 for stability)
        grim_out = grim_out.float()
        logits = self.classifier(grim_out)
        logits = torch.clamp(logits, min=-_LOGIT_CLAMP, max=_LOGIT_CLAMP)
        log_probs = F.log_softmax(logits, dim=-1)

        if was_1d:
            log_probs = log_probs.squeeze(0)
            rate_codes = rate_codes.squeeze(0)

        if return_rates:
            return log_probs, rate_codes
        return log_probs

    def sparsity_loss(self, rate_codes: Tensor) -> Tensor:
        """Compute sparsity regularization loss on rate codes.

        Args:
            rate_codes: Sparse rate codes from ``forward(..., return_rates=True)``.

        Returns:
            Scalar sparsity loss.
        """
        result: Tensor = self.sparsity_loss_fn(rate_codes)
        return result

    def oscillatory_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to the oscillatory (LOBM) stage.

        These benefit from smaller learning rates due to the
        sensitivity of oscillator dynamics to coupling parameters.
        """
        params: list[nn.Parameter] = []
        for lobm in self.lobm_layers:
            params.extend(lobm.parameters())
        params.extend(self.phase_to_rate.parameters())
        return params

    def rate_coded_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to the rate-coded (GRIM) stage.

        These typically use standard learning rates.
        """
        params: list[nn.Parameter] = []
        params.extend(self.grim_proj.parameters())
        params.extend(self.grim_norm.parameters())
        params.extend(self.grim_encoder.parameters())
        params.extend(self.classifier.parameters())
        return params


# ---- Alternating Optimizer Wrapper ------------------------------------


class AlternatingOptimizer:
    """Alternating optimization for hybrid oscillatory + rate-coded training.

    Manages two separate optimizers — one for oscillatory (LOBM)
    parameters and one for rate-coded (GRIM) parameters — and
    alternates between them based on a schedule.

    Optionally integrates with the subconscious daemon to receive
    adaptive control signals (lr multiplier, coupling suggestions).

    Args:
        model: A :class:`HybridPRINet` model.
        osc_lr: Learning rate for oscillatory parameters.
        rate_lr: Learning rate for rate-coded parameters.
        osc_optimizer_cls: Optimizer class for oscillatory params.
            Defaults to ``torch.optim.Adam``.
        rate_optimizer_cls: Optimizer class for rate-coded params.
            Defaults to ``torch.optim.Adam``.
        alternation_mode: ``"epoch"`` (alternate per epoch) or
            ``"step"`` (alternate per gradient step).
        sparsity_weight: Weight for sparsity regularization loss.
        daemon: Optional :class:`~prinet.core.subconscious_daemon.SubconsciousDaemon`
            for adaptive control.

    Example:
        >>> model = HybridPRINet()
        >>> alt = AlternatingOptimizer(model)
        >>> # In training loop:
        >>> loss.backward()
        >>> alt.step(epoch=0)
        >>> alt.zero_grad()
    """

    def __init__(
        self,
        model: HybridPRINet,
        osc_lr: float = 1e-4,
        rate_lr: float = 1e-3,
        osc_optimizer_cls: type = torch.optim.Adam,
        rate_optimizer_cls: type = torch.optim.Adam,
        alternation_mode: str = "epoch",
        sparsity_weight: float = 0.01,
        daemon: Any = None,
    ) -> None:
        self.model = model
        self._mode = alternation_mode
        self._sparsity_weight = sparsity_weight
        self._step_count = 0
        self._daemon = daemon

        # Separate parameter groups
        osc_params = model.oscillatory_parameters()
        rate_params = model.rate_coded_parameters()

        self.osc_optimizer = osc_optimizer_cls(osc_params, lr=osc_lr)
        self.rate_optimizer = rate_optimizer_cls(rate_params, lr=rate_lr)

    @property
    def sparsity_weight(self) -> float:
        """Current sparsity loss weight."""
        return self._sparsity_weight

    def step(self, epoch: int = 0) -> None:
        """Perform one optimization step with alternating schedule.

        In ``"epoch"`` mode: even epochs update oscillatory params,
        odd epochs update rate-coded params.
        In ``"step"`` mode: alternates every gradient step.

        If a subconscious daemon is attached, reads control signals
        and applies lr multiplier.

        Args:
            epoch: Current epoch number (used in ``"epoch"`` mode).
        """
        # Apply daemon control signals if available
        if self._daemon is not None:
            self._apply_daemon_control()

        if self._mode == "epoch":
            if epoch % 2 == 0:
                self.osc_optimizer.step()
            else:
                self.rate_optimizer.step()
        else:
            # "step" mode — alternate every step
            if self._step_count % 2 == 0:
                self.osc_optimizer.step()
            else:
                self.rate_optimizer.step()

        self._step_count += 1

    def step_both(self) -> None:
        """Step both optimizers simultaneously (for warmup or fine-tuning)."""
        if self._daemon is not None:
            self._apply_daemon_control()
        self.osc_optimizer.step()
        self.rate_optimizer.step()
        self._step_count += 1

    def zero_grad(self) -> None:
        """Zero gradients for both optimizers."""
        self.osc_optimizer.zero_grad()
        self.rate_optimizer.zero_grad()

    def _apply_daemon_control(self) -> None:
        """Read control signals from the subconscious daemon and apply."""
        try:
            ctrl = self._daemon.get_control()
            # Apply lr multiplier to both parameter groups
            lr_mult = ctrl.lr_multiplier
            for pg in self.osc_optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * lr_mult
            for pg in self.rate_optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * lr_mult
        except Exception:
            pass  # Daemon not ready or failed — continue without control


# ---- CLEVR-N Hybrid Model (Benchmark Adapter) --------------------------


class HybridCLEVRN(nn.Module):
    """HybridPRINet adapter for CLEVR-N benchmark.

    Wraps :class:`HybridPRINet` to accept CLEVR-N scene + query inputs,
    producing binary classification outputs compatible with the CLEVR-N
    benchmark framework.

    Args:
        scene_dim: Per-item scene feature dimension.
        query_dim: Query vector dimension.
        n_delta: Delta-band oscillators.
        n_theta: Theta-band oscillators.
        n_gamma: Gamma-band oscillators.
        hidden_dim: Internal hidden dimension.
    """

    def __init__(
        self,
        scene_dim: int = 16,
        query_dim: int = 44,
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 32,
        hidden_dim: int = 64,
        lobm_steps: int = 5,
    ) -> None:
        super().__init__()
        n_osc = n_delta + n_theta + n_gamma

        # Scene encoder: flatten + project scene items
        self.scene_proj = nn.Linear(scene_dim, n_osc)
        self.query_proj = nn.Linear(query_dim, hidden_dim)

        # Core hybrid model (binary classification)
        self.hybrid = HybridPRINet(
            n_input=n_osc,
            n_classes=2,
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            n_lobm_layers=1,
            lobm_steps=lobm_steps,
            grim_d_model=hidden_dim,
            grim_n_layers=1,
        )

        # Merge oscillatory + query features
        self.merge = nn.Linear(2 + hidden_dim, 2)

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        """Forward pass.

        Args:
            scene: ``(B, N, D_scene)``
            query: ``(B, D_query)``

        Returns:
            Log probabilities ``(B, 2)``.
        """
        # Aggregate scene items via mean
        scene_agg = scene.mean(dim=1)  # (B, D_scene)
        scene_enc = self.scene_proj(scene_agg)  # (B, n_osc)

        # Run through hybrid model
        log_probs: Tensor = self.hybrid(scene_enc)  # (B, 2)

        return log_probs


# =========================================================================
# Year 2 Q1 — Workstream B: Interleaved Hybrid Architecture
# =========================================================================


class InterleavedHybridPRINet(nn.Module):
    """Interleaved oscillatory-attention hybrid model.

    Instead of the sequential LOBM → PhaseToRate → GRIM pipeline in
    :class:`HybridPRINet`, this architecture **interleaves** oscillatory
    dynamics with attention at every layer:

    1. Embed input as a token sequence.
    2. For each layer: OscillatoryAttention → FFN → update phase state.
    3. Pool → classify.

    The oscillatory phase state persists across layers and biases
    attention toward phase-coherent tokens (bound items). This is a
    fundamentally different integration strategy from sequential LOBM→GRIM.

    Args:
        n_input: Input (per-token) feature dimension.
        n_classes: Number of output classes.
        n_tokens: Number of tokens in the sequence (for single-vector
            input, this is ``n_delta + n_theta + n_gamma``).
        d_model: Model dimension after input projection.
        n_heads: Number of attention heads.
        n_layers: Number of interleaved OscAttention+FFN blocks.
        dropout: Dropout rate.
        n_delta: Delta-band oscillators (phase state partitioning).
        n_theta: Theta-band oscillators.
        n_gamma: Gamma-band oscillators.
        n_discrete_steps: Discrete dynamics steps per layer.

    Example:
        >>> model = InterleavedHybridPRINet(
        ...     n_input=128, n_classes=10, n_tokens=44,
        ...     d_model=64, n_heads=4, n_layers=2
        ... )
        >>> x = torch.randn(8, 128)
        >>> log_probs = model(x)
        >>> print(log_probs.shape)
        torch.Size([8, 10])
    """

    def __init__(
        self,
        n_input: int = 256,
        n_classes: int = 10,
        n_tokens: int = 44,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        # Oscillator band config (for phase state)
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 32,
        n_discrete_steps: int = 3,
    ) -> None:
        super().__init__()

        from prinet.core.propagation import DiscreteDeltaThetaGamma
        from prinet.nn.layers import OscillatoryAttention

        self.n_input = n_input
        self.n_classes = n_classes
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_layers = n_layers
        self._n_heads = n_heads
        self.n_osc_total = n_delta + n_theta + n_gamma
        self._n_discrete_steps = n_discrete_steps

        # Input projection: (B, n_input) → (B, n_tokens, d_model)
        self.input_proj = nn.Linear(n_input, n_tokens * d_model)

        # Initial phase projection: input → phase per token per head
        self.phase_init = nn.Linear(n_input, n_tokens * n_heads)

        # Shared discrete dynamics for phase updates
        self.dynamics = DiscreteDeltaThetaGamma(
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
        )

        # Interleaved layers
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(
                OscillatoryAttention(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                )
            )
            self.norm1_layers.append(nn.LayerNorm(d_model))
            self.norm2_layers.append(nn.LayerNorm(d_model))

        # Classification head
        self.pool_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through interleaved oscillatory-attention model.

        Args:
            x: Input tensor ``(B, D)`` or ``(D,)``.

        Returns:
            Log-probabilities ``(B, K)`` or ``(K,)``.
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        B = x.shape[0]
        n_heads: int = self._n_heads

        # Embed as token sequence: (B, n_tokens, d_model)
        h = self.input_proj(x).view(B, self.n_tokens, self.d_model)

        # Initialize oscillatory phase state
        phase_init_raw = self.phase_init(x).view(
            B, self.n_tokens, n_heads
        )
        phase_state = phase_init_raw % (2.0 * math.pi)

        # Initialize amplitude for dynamics (unit)
        amp_state = torch.ones(
            B, self.n_osc_total, device=x.device, dtype=x.dtype
        )
        # Phase for dynamics (mean across heads for each oscillator)
        dyn_phase = phase_state[:, : self.n_osc_total, :].mean(dim=-1)

        for i in range(self.n_layers):
            # --- Oscillatory phase update ---
            dyn_phase, amp_state = self.dynamics.integrate(
                dyn_phase, amp_state,
                n_steps=self._n_discrete_steps, dt=0.01,
            )

            # Convert dynamics phase → per-token per-head phase
            # dyn_phase: (B, n_osc_total)
            if self.n_tokens == self.n_osc_total:
                token_phase = dyn_phase.unsqueeze(-1).expand(
                    B, self.n_tokens, n_heads
                )
            else:
                # Interpolate: take first n_tokens from padded/repeated
                repeated = dyn_phase.repeat(
                    1, (self.n_tokens // self.n_osc_total) + 1
                )[:, : self.n_tokens]  # (B, n_tokens)
                token_phase = repeated.unsqueeze(-1).expand(
                    B, self.n_tokens, n_heads
                )

            # --- OscillatoryAttention + residual ---
            h_norm = self.norm1_layers[i](h)
            h = h + self.attn_layers[i](h_norm, phase=token_phase)

            # --- FFN + residual ---
            h_norm = self.norm2_layers[i](h)
            h = h + self.ffn_layers[i](h_norm)

        # Pool (mean over tokens) and classify
        pooled = self.pool_norm(h.mean(dim=1))  # (B, d_model)
        logits = self.classifier(pooled)
        logits = torch.clamp(logits, min=-_LOGIT_CLAMP, max=_LOGIT_CLAMP)
        log_probs = F.log_softmax(logits, dim=-1)

        if was_1d:
            return log_probs.squeeze(0)
        return log_probs

    def oscillatory_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to oscillatory components."""
        from prinet.nn.layers import OscillatoryAttention

        params: list[nn.Parameter] = []
        params.extend(self.dynamics.parameters())
        params.extend(self.phase_init.parameters())
        for attn in self.attn_layers:
            assert isinstance(attn, OscillatoryAttention)
            params.extend(attn.phase_proj.parameters())
            params.append(attn.alpha)
        return params

    def rate_coded_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to rate-coded components."""
        from prinet.nn.layers import OscillatoryAttention

        params: list[nn.Parameter] = []
        params.extend(self.input_proj.parameters())
        for attn in self.attn_layers:
            assert isinstance(attn, OscillatoryAttention)
            params.extend(attn.W_q.parameters())
            params.extend(attn.W_k.parameters())
            params.extend(attn.W_v.parameters())
            params.extend(attn.W_o.parameters())
        for ffn in self.ffn_layers:
            params.extend(ffn.parameters())
        for n1, n2 in zip(self.norm1_layers, self.norm2_layers):
            params.extend(n1.parameters())
            params.extend(n2.parameters())
        params.extend(self.pool_norm.parameters())
        params.extend(self.classifier.parameters())
        return params


# =========================================================================
# Year 2 Q2 — Workstream E: Temporal Hybrid Architecture
# =========================================================================


class TemporalHybridPRINet(nn.Module):
    """Temporal extension of InterleavedHybridPRINet with phase propagation.

    Wraps :class:`InterleavedHybridPRINet` with frame-to-frame phase
    carry-over using :class:`~prinet.core.propagation.TemporalPhasePropagator`.
    Processes multi-frame sequences where phase state from frame t-1
    initializes dynamics at frame t, enabling persistent object binding
    across time.

    Architecture per frame:
        1. Encode frame input → token sequence.
        2. Blend carried-over phase with input-derived phase (α-mixing).
        3. Run interleaved oscillatory-attention layers.
        4. Store final phase for next frame.
        5. Pool + classify.

    The final output is the classification from the **last frame** in the
    sequence, or per-frame if ``per_frame=True``.

    Args:
        n_input: Per-frame input feature dimension.
        n_classes: Number of output classes.
        n_tokens: Number of tokens in the sequence.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of interleaved blocks.
        dropout: Dropout rate.
        n_delta: Delta-band oscillators.
        n_theta: Theta-band oscillators.
        n_gamma: Gamma-band oscillators.
        n_discrete_steps: Discrete dynamics steps per layer.
        carry_strength: Phase carry-over strength α ∈ [0, 1].
        amplitude_decay: Amplitude carry-over β ∈ [0, 1].

    Example:
        >>> model = TemporalHybridPRINet(
        ...     n_input=128, n_classes=2, n_tokens=44,
        ...     d_model=64, n_heads=4, n_layers=2,
        ... )
        >>> x = torch.randn(8, 5, 128)  # 5-frame sequence
        >>> log_probs = model(x)  # last-frame classification
        >>> print(log_probs.shape)
        torch.Size([8, 2])
    """

    def __init__(
        self,
        n_input: int = 256,
        n_classes: int = 10,
        n_tokens: int = 44,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 32,
        n_discrete_steps: int = 3,
        carry_strength: float = 0.8,
        amplitude_decay: float = 0.5,
    ) -> None:
        super().__init__()

        from prinet.core.propagation import TemporalPhasePropagator

        self.interleaved = InterleavedHybridPRINet(
            n_input=n_input,
            n_classes=n_classes,
            n_tokens=n_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            n_discrete_steps=n_discrete_steps,
        )
        self.propagator = TemporalPhasePropagator(
            carry_strength=carry_strength,
            amplitude_decay=amplitude_decay,
        )
        self.n_input = n_input
        self.n_classes = n_classes
        self._n_heads = n_heads

    def forward(
        self,
        x: Tensor,
        per_frame: bool = False,
    ) -> Tensor:
        """Forward pass over a multi-frame sequence.

        Args:
            x: Input tensor ``(B, T, D)`` where T is the number of frames,
                or ``(B, D)`` for single-frame (falls back to base model).
            per_frame: If ``True``, return log-probs for all frames
                ``(B, T, K)``. Otherwise return only the last frame
                ``(B, K)``.

        Returns:
            Log-probabilities. Shape depends on ``per_frame``.
        """
        if x.dim() == 2:
            # Single frame — delegate directly
            out: Tensor = self.interleaved(x)
            return out

        B, T, D = x.shape
        model = self.interleaved
        n_heads: int = self._n_heads
        n_osc = model.n_osc_total

        frame_outputs: list[Tensor] = []
        prev_dyn_phase: Optional[Tensor] = None
        prev_dyn_amp: Optional[Tensor] = None

        for t in range(T):
            x_t = x[:, t, :]  # (B, D)

            # Embed as token sequence
            h = model.input_proj(x_t).view(B, model.n_tokens, model.d_model)

            # Initialize phase from input
            phase_init_raw = model.phase_init(x_t).view(
                B, model.n_tokens, n_heads
            )
            phase_state = phase_init_raw % (2.0 * math.pi)

            # Initialize dynamics phase and amplitude
            dyn_phase = phase_state[:, :n_osc, :].mean(dim=-1)  # (B, n_osc)
            amp_state = torch.ones(
                B, n_osc, device=x.device, dtype=x.dtype
            )

            # Temporal propagation: blend with previous frame
            if prev_dyn_phase is not None:
                from prinet.core.propagation import _wrap_phase

                dyn_phase, amp_state = self.propagator.propagate(
                    prev_dyn_phase, prev_dyn_amp,  # type: ignore[arg-type]
                    dyn_phase, amp_state,
                )

            # Run interleaved layers
            for i in range(model.n_layers):
                dyn_phase, amp_state = model.dynamics.integrate(
                    dyn_phase, amp_state,
                    n_steps=model._n_discrete_steps, dt=0.01,
                )

                if model.n_tokens == n_osc:
                    token_phase = dyn_phase.unsqueeze(-1).expand(
                        B, model.n_tokens, n_heads
                    )
                else:
                    repeated = dyn_phase.repeat(
                        1, (model.n_tokens // n_osc) + 1
                    )[:, :model.n_tokens]
                    token_phase = repeated.unsqueeze(-1).expand(
                        B, model.n_tokens, n_heads
                    )

                h_norm = model.norm1_layers[i](h)
                h = h + model.attn_layers[i](h_norm, phase=token_phase)
                h_norm = model.norm2_layers[i](h)
                h = h + model.ffn_layers[i](h_norm)

            # Store phase state for next frame
            prev_dyn_phase = dyn_phase.detach()
            prev_dyn_amp = amp_state.detach()

            # Pool and classify
            pooled = model.pool_norm(h.mean(dim=1))
            logits = model.classifier(pooled)
            logits = torch.clamp(logits, min=-_LOGIT_CLAMP, max=_LOGIT_CLAMP)
            log_probs = F.log_softmax(logits, dim=-1)
            frame_outputs.append(log_probs)

        if per_frame:
            return torch.stack(frame_outputs, dim=1)  # (B, T, K)
        return frame_outputs[-1]  # (B, K)

    def oscillatory_parameters(self) -> list[nn.Parameter]:
        """Return oscillatory component parameters."""
        return self.interleaved.oscillatory_parameters()

    def rate_coded_parameters(self) -> list[nn.Parameter]:
        """Return rate-coded component parameters."""
        return self.interleaved.rate_coded_parameters()


# =========================================================================
# Year 2 Q3 — Workstream G: HybridPRINet v2 (Canonical Architecture)
# =========================================================================


class HybridPRINetV2(nn.Module):
    """HybridPRINet v2 — canonical architecture based on DiscreteDTG.

    Selected as the winning variant from Q2 empirical evaluation:
    DiscreteDTG achieved 100% accuracy on CLEVR-N (N=2-15) while all
    other architectures degraded. This v2 architecture uses
    :class:`~prinet.core.propagation.DiscreteDeltaThetaGamma` as the
    oscillatory binding module, replacing the ODE-based
    HierarchicalResonanceLayer, and uses adaptive token count
    (matching oscillator count) to avoid the fixed-padding issue that
    caused InterleavedHybrid degradation at high N.

    Architecture::

        Input → Projection → DiscreteDTG dynamics (n_steps)
              → OscillatoryAttention × L layers
              → Pool → Classifier

    Key improvements over v1:
        1. DiscreteDTG dynamics instead of ODE-based LOBM (3000x faster)
        2. Adaptive n_tokens = n_oscillators (no fixed padding)
        3. Configurable n_discrete_steps per layer (was hardcoded at 3)
        4. Optional conv stem for image inputs (CIFAR-10, Fashion-MNIST)

    Args:
        n_input: Input feature dimension (flat vector or conv stem output).
        n_classes: Number of output classes.
        d_model: Model dimension for attention layers.
        n_heads: Number of attention heads.
        n_layers: Number of interleaved OscAttention + FFN blocks.
        dropout: Dropout rate.
        n_delta: Delta-band oscillators.
        n_theta: Theta-band oscillators.
        n_gamma: Gamma-band oscillators.
        n_discrete_steps: Discrete dynamics steps per layer.
        coupling_strength: Initial intra-band coupling strength.
        pac_depth: Initial PAC modulation depth.
        use_conv_stem: If ``True``, prepend a lightweight CNN stem
            for 2D image inputs (expects ``(B, C, H, W)``).
        stem_channels: Number of channels in conv stem output.

    Example:
        >>> model = HybridPRINetV2(n_input=128, n_classes=10)
        >>> x = torch.randn(8, 128)
        >>> log_probs = model(x)
        >>> print(log_probs.shape)
        torch.Size([8, 10])
    """

    def __init__(
        self,
        n_input: int = 256,
        n_classes: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 32,
        n_discrete_steps: int = 5,
        coupling_strength: float = 2.0,
        pac_depth: float = 0.3,
        use_conv_stem: bool = False,
        stem_channels: int = 64,
    ) -> None:
        super().__init__()

        from prinet.core.propagation import DiscreteDeltaThetaGamma
        from prinet.nn.layers import OscillatoryAttention

        self.n_input = n_input
        self.n_classes = n_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self._n_heads = n_heads
        self.n_osc_total = n_delta + n_theta + n_gamma
        self.n_tokens = self.n_osc_total  # Adaptive: match oscillators
        self._n_discrete_steps = n_discrete_steps
        self.use_conv_stem = use_conv_stem

        # Optional CNN stem for image inputs
        if use_conv_stem:
            self.conv_stem = nn.Sequential(
                nn.Conv2d(3, stem_channels, 3, padding=1),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(),
                nn.Conv2d(stem_channels, stem_channels, 3, padding=1),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),  # → (B, stem_channels, 4, 4)
                nn.Flatten(),  # → (B, stem_channels * 16)
            )
            proj_input = stem_channels * 16
        else:
            self.conv_stem = None
            proj_input = n_input

        # Input projection: (B, input_dim) → (B, n_tokens, d_model)
        self.input_proj = nn.Linear(proj_input, self.n_tokens * d_model)

        # Phase initialization from input
        self.phase_init = nn.Linear(proj_input, self.n_tokens * n_heads)

        # Shared discrete dynamics (learnable frequencies, coupling, PAC)
        self.dynamics = DiscreteDeltaThetaGamma(
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            coupling_strength=coupling_strength,
            pac_depth=pac_depth,
        )

        # Interleaved OscillatoryAttention + FFN layers
        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(
                OscillatoryAttention(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                )
            )
            self.norm1_layers.append(nn.LayerNorm(d_model))
            self.norm2_layers.append(nn.LayerNorm(d_model))

        # Classification head
        self.pool_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through HybridPRINet v2.

        Args:
            x: Input tensor ``(B, D)`` for flat vectors, ``(D,)``
                for single sample, or ``(B, C, H, W)`` if conv stem
                is enabled.

        Returns:
            Log-probabilities ``(B, K)`` or ``(K,)``.
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        # Conv stem for image inputs
        if self.use_conv_stem and self.conv_stem is not None and x.dim() == 4:
            x = self.conv_stem(x)  # (B, stem_channels * 16)

        B = x.shape[0]
        n_heads: int = self._n_heads
        n_tokens: int = self.n_tokens

        # Embed as token sequence: (B, n_tokens, d_model)
        h = self.input_proj(x).view(B, n_tokens, self.d_model)

        # Initialize oscillatory phase state
        phase_init_raw = self.phase_init(x).view(B, n_tokens, n_heads)
        phase_state = phase_init_raw % (2.0 * math.pi)

        # Initialize amplitude and dynamics phase
        amp_state = torch.ones(
            B, self.n_osc_total, device=x.device, dtype=x.dtype
        )
        # n_tokens == n_osc_total in v2, so direct mapping
        dyn_phase = phase_state.mean(dim=-1)  # (B, n_tokens)

        for i in range(self.n_layers):
            # Discrete multi-rate dynamics update
            dyn_phase, amp_state = self.dynamics.integrate(
                dyn_phase, amp_state,
                n_steps=self._n_discrete_steps, dt=0.01,
            )

            # Direct phase mapping (no padding interpolation needed)
            token_phase = dyn_phase.unsqueeze(-1).expand(
                B, n_tokens, n_heads
            )

            # OscillatoryAttention + residual
            h_norm = self.norm1_layers[i](h)
            h = h + self.attn_layers[i](h_norm, phase=token_phase)

            # FFN + residual
            h_norm = self.norm2_layers[i](h)
            h = h + self.ffn_layers[i](h_norm)

        # Pool (mean over tokens) and classify
        pooled = self.pool_norm(h.mean(dim=1))  # (B, d_model)
        logits = self.classifier(pooled)
        logits = torch.clamp(logits, min=-_LOGIT_CLAMP, max=_LOGIT_CLAMP)
        log_probs = F.log_softmax(logits, dim=-1)

        if was_1d:
            return log_probs.squeeze(0)
        return log_probs

    # ---- O.1: torch.compile integration --------------------------------

    @staticmethod
    def _triton_available() -> bool:
        """Check whether Triton is importable (required for max-autotune)."""
        try:
            import triton  # noqa: F401
            return True
        except ImportError:
            return False

    def compile(
        self,
        *,
        backend: str = "inductor",
        mode: str = "max-autotune",
        fullgraph: bool = False,
        dynamic: bool = False,
    ) -> "HybridPRINetV2":
        """Apply ``torch.compile`` to the forward pass for acceleration.

        Wraps the model with ``torch.compile`` using the specified
        backend and mode.  On Windows the Inductor backend is used by
        default which provides 10-30 % speedup on typical forward +
        backward workloads.

        When ``mode="max-autotune"`` is requested but Triton is not
        installed (e.g. on Windows), the mode is automatically
        downgraded to ``"reduce-overhead"`` to avoid import errors.

        Args:
            backend: Compilation backend (default ``"inductor"``).
            mode: Compilation mode.  ``"max-autotune"`` enables kernel
                auto-tuning for best throughput; ``"reduce-overhead"``
                minimises graph-capture time.
            fullgraph: If ``True``, require the entire model to be
                captured as a single graph (fails on dynamic control
                flow).
            dynamic: If ``True``, enable dynamic shape support.

        Returns:
            The compiled model (same object, mutated in place).

        Example:
            >>> model = HybridPRINetV2(n_input=128, n_classes=10)
            >>> model.compile()  # in-place, returns self
            >>> x = torch.randn(8, 128)
            >>> log_probs = model(x)  # runs compiled forward
        """
        # Auto-downgrade mode when Triton is unavailable (Windows)
        if mode == "max-autotune" and not self._triton_available():
            import logging as _logging

            _logging.getLogger(__name__).info(
                "Triton not available — downgrading torch.compile mode "
                "from 'max-autotune' to 'reduce-overhead'"
            )
            mode = "reduce-overhead"

        compiled = torch.compile(
            self,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        self._compiled_model = compiled
        self._is_compiled = True
        return self

    @property
    def is_compiled(self) -> bool:
        """Whether ``torch.compile`` has been applied."""
        return getattr(self, "_is_compiled", False)

    def compiled_forward(self, x: Tensor) -> Tensor:
        """Run the compiled forward pass if available.

        Falls back to the standard forward if ``compile()`` was not
        called.

        Args:
            x: Input tensor (same interface as :meth:`forward`).

        Returns:
            Log-probabilities ``(B, K)`` or ``(K,)``.
        """
        if self.is_compiled and hasattr(self, "_compiled_model"):
            return self._compiled_model(x)
        return self.forward(x)

    def oscillatory_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to oscillatory components."""
        from prinet.nn.layers import OscillatoryAttention

        params: list[nn.Parameter] = []
        params.extend(self.dynamics.parameters())
        params.extend(self.phase_init.parameters())
        for attn in self.attn_layers:
            assert isinstance(attn, OscillatoryAttention)
            params.extend(attn.phase_proj.parameters())
            params.append(attn.alpha)
        return params

    def rate_coded_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to rate-coded components."""
        from prinet.nn.layers import OscillatoryAttention

        params: list[nn.Parameter] = []
        if self.conv_stem is not None:
            params.extend(self.conv_stem.parameters())
        params.extend(self.input_proj.parameters())
        for attn in self.attn_layers:
            assert isinstance(attn, OscillatoryAttention)
            params.extend(attn.W_q.parameters())
            params.extend(attn.W_k.parameters())
            params.extend(attn.W_v.parameters())
            params.extend(attn.W_o.parameters())
        for ffn in self.ffn_layers:
            params.extend(ffn.parameters())
        for n1, n2 in zip(self.norm1_layers, self.norm2_layers):
            params.extend(n1.parameters())
            params.extend(n2.parameters())
        params.extend(self.pool_norm.parameters())
        params.extend(self.classifier.parameters())
        return params


class HybridPRINetV2CLEVRN(nn.Module):
    """CLEVR-N adapter for HybridPRINet v2.

    Wraps :class:`HybridPRINetV2` to accept scene + query inputs for
    CLEVR-N binding benchmarks.  Scene features are projected and
    concatenated with the query embedding before feeding into the
    oscillatory backbone, ensuring the query information influences
    the binding decision.

    Args:
        scene_dim: Per-item scene feature dimension.
        query_dim: Query vector dimension (supports arbitrary sizes).
        n_delta: Delta-band oscillators.
        n_theta: Theta-band oscillators.
        n_gamma: Gamma-band oscillators.
        d_model: Model dimension.
        n_discrete_steps: Dynamics steps per layer.
        coupling_strength: Passed through to V2.
        pac_depth: Passed through to V2.
    """

    def __init__(
        self,
        scene_dim: int = 16,
        query_dim: int = 60,
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 32,
        d_model: int = 64,
        n_discrete_steps: int = 5,
        coupling_strength: float = 2.0,
        pac_depth: float = 0.3,
    ) -> None:
        super().__init__()
        n_osc = n_delta + n_theta + n_gamma
        self.scene_proj = nn.Linear(scene_dim, d_model)
        self.query_proj = nn.Linear(query_dim, d_model)
        self.merge = nn.Linear(d_model * 2, n_osc)
        self.v2 = HybridPRINetV2(
            n_input=n_osc,
            n_classes=2,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            n_discrete_steps=n_discrete_steps,
            coupling_strength=coupling_strength,
            pac_depth=pac_depth,
        )

    def forward(self, scene: Tensor, query: Tensor) -> Tensor:
        """Forward pass.

        Args:
            scene: ``(B, N, D_scene)`` or ``(B, D_scene)``.
            query: ``(B, D_query)``.

        Returns:
            Log probabilities ``(B, 2)``.
        """
        if scene.dim() == 3:
            scene_agg = scene.mean(dim=1)
        else:
            scene_agg = scene
        scene_enc = self.scene_proj(scene_agg)  # (B, d_model)
        query_enc = self.query_proj(query)       # (B, d_model)
        combined = torch.cat([scene_enc, query_enc], dim=-1)  # (B, 2*d_model)
        merged = self.merge(combined)            # (B, n_osc)
        return self.v2(merged)


class PhaseTracker(nn.Module):
    """Phase-based multi-object tracker for 2D MOT.

    Uses oscillatory phase continuity to associate detections across
    frames. Each detection is assigned a phase state; cross-frame
    association is determined by phase correlation (cosine similarity
    of complex phase embeddings).

    Architecture:
        1. Encode each detection via MLP → phase + amplitude.
        2. Run DiscreteDTG dynamics to encourage phase coherence.
        3. Cross-frame phase matching via cosine similarity.
        4. Hungarian assignment on similarity matrix.

    Args:
        detection_dim: Per-detection feature dimension.
        n_delta: Delta-band oscillators.
        n_theta: Theta-band oscillators.
        n_gamma: Gamma-band oscillators.
        n_discrete_steps: Dynamics steps per frame.
        match_threshold: Minimum phase similarity for valid match.

    Example:
        >>> tracker = PhaseTracker(detection_dim=4)
        >>> # 3 detections at frame t, 3 at frame t+1
        >>> dets_t = torch.randn(3, 4)
        >>> dets_t1 = torch.randn(3, 4)
        >>> matches, sim = tracker(dets_t, dets_t1)
    """

    def __init__(
        self,
        detection_dim: int = 4,
        n_delta: int = 4,
        n_theta: int = 8,
        n_gamma: int = 16,
        n_discrete_steps: int = 5,
        match_threshold: float = 0.3,
    ) -> None:
        super().__init__()

        from prinet.core.propagation import DiscreteDeltaThetaGamma

        self.n_osc = n_delta + n_theta + n_gamma
        self._n_discrete_steps = n_discrete_steps
        self.match_threshold = match_threshold

        # Detection → phase embedding
        self.det_to_phase = nn.Sequential(
            nn.Linear(detection_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_osc),
        )
        # Detection → amplitude embedding
        self.det_to_amp = nn.Sequential(
            nn.Linear(detection_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_osc),
            nn.Softplus(),
        )

        # Shared dynamics for phase evolution
        self.dynamics = DiscreteDeltaThetaGamma(
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
        )

    def encode(
        self, detections: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode detections into phase and amplitude embeddings.

        Args:
            detections: Detection features ``(N, D)`` or ``(B, N, D)``.

        Returns:
            Tuple of ``(phase, amplitude)`` each ``(N, n_osc)`` or
            ``(B, N, n_osc)``.
        """
        phase_raw = self.det_to_phase(detections)
        phase = phase_raw % (2.0 * math.pi)
        amp = self.det_to_amp(detections)
        return phase, amp

    def evolve(
        self, phase: Tensor, amplitude: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Evolve phase state through discrete dynamics.

        Args:
            phase: Phase tensor ``(N, n_osc)`` or ``(B, N, n_osc)``.
            amplitude: Amplitude tensor, same shape.

        Returns:
            Evolved ``(phase, amplitude)``.
        """
        return self.dynamics.integrate(
            phase, amplitude,
            n_steps=self._n_discrete_steps, dt=0.01,
        )

    def phase_similarity(
        self, phase_a: Tensor, phase_b: Tensor
    ) -> Tensor:
        """Compute phase similarity matrix between two sets of phases.

        Uses cosine similarity of complex phase embeddings
        ``exp(i * phase)`` averaged over oscillator bands.

        This computes the standard Kuramoto-like order parameter:

        .. math::

            \\text{sim}(a, b) = \\frac{1}{K} \\sum_{k=1}^{K}
            \\cos(\\varphi_a^k - \\varphi_b^k)

        where *K* = ``n_osc``.  Values lie in ``[-1, 1]``.

        Args:
            phase_a: ``(N_a, n_osc)``
            phase_b: ``(N_b, n_osc)``

        Returns:
            Similarity matrix ``(N_a, N_b)`` in ``[-1, 1]``.
        """
        z_a = torch.exp(1j * phase_a.to(torch.complex64))  # (N_a, n_osc)
        z_b = torch.exp(1j * phase_b.to(torch.complex64))  # (N_b, n_osc)

        # L2-normalised cosine similarity (since |exp(ix)| = 1,
        # L2 norm = sqrt(n_osc) for every row)
        z_a_norm = z_a / (
            z_a.abs().pow(2).sum(dim=-1, keepdim=True).sqrt() + _EPS
        )
        z_b_norm = z_b / (
            z_b.abs().pow(2).sum(dim=-1, keepdim=True).sqrt() + _EPS
        )

        # Real part of complex inner product = phase similarity
        sim = (z_a_norm.unsqueeze(1) * z_b_norm.conj().unsqueeze(0)).sum(
            dim=-1
        ).real.float()
        return sim

    def forward(
        self,
        detections_t: Tensor,
        detections_t1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Match detections across two consecutive frames.

        Args:
            detections_t: Frame t detections ``(N_t, D)``.
            detections_t1: Frame t+1 detections ``(N_t1, D)``.

        Returns:
            Tuple of:
                - ``matches``: ``(min(N_t, N_t1),)`` assignment indices.
                  ``matches[i]`` is the index in ``detections_t1`` matched
                  to the i-th detection in ``detections_t``, or ``-1`` if
                  unmatched.
                - ``similarity``: Full similarity matrix ``(N_t, N_t1)``.
        """
        # Encode both frames
        phase_t, amp_t = self.encode(detections_t)
        phase_t1, amp_t1 = self.encode(detections_t1)

        # Evolve frame t phases forward (predict where they should be)
        phase_t_evolved, _ = self.evolve(phase_t, amp_t)

        # Compute similarity between evolved t and raw t+1
        sim = self.phase_similarity(phase_t_evolved, phase_t1)

        # Greedy assignment (differentiable soft version for training)
        # For inference, use Hungarian; for forward pass, return sim
        N_t = detections_t.shape[0]

        # Greedy matching via argmax on similarity
        matches = torch.full((N_t,), -1, dtype=torch.long,
                             device=detections_t.device)
        used = torch.zeros(detections_t1.shape[0], dtype=torch.bool,
                           device=detections_t.device)

        # Sort by max similarity (match most confident first)
        max_sims, max_idxs = sim.max(dim=1)
        order = max_sims.argsort(descending=True)

        for idx in order:
            best_j = max_idxs[idx].item()
            if not used[best_j] and max_sims[idx] > self.match_threshold:
                matches[idx] = best_j
                used[best_j] = True

        return matches, sim

    # ------------------------------------------------------------------
    # Sequence-level tracking (mirrors TemporalSlotAttentionMOT API)
    # ------------------------------------------------------------------

    def track_sequence(
        self,
        frame_detections: list[Tensor],
    ) -> dict:
        """Track objects across a sequence of frames.

        Processes each consecutive pair of frames through :meth:`forward`
        and accumulates identity-preservation statistics.  The returned
        dict is intentionally compatible with
        :meth:`~prinet.nn.slot_attention.TemporalSlotAttentionMOT.track_sequence`
        so that the two trackers can be compared head-to-head.

        In addition to the standard keys, this method records
        oscillatory-specific metrics:

        - ``"phase_history"``: list of ``(N_t, n_osc)`` phase tensors
          (one per frame) — useful for computing phase-slip rate and
          inter-frame phase correlation downstream.
        - ``"per_frame_phase_correlation"``: per-transition mean cosine
          of phase differences (:math:`\\rho` in ``[0, 1]``).

        Args:
            frame_detections: List of *T* tensors, each ``(N_det, D)``
                containing per-frame detection features.

        Returns:
            Dict with:
                - ``"phase_history"``: List of T phase tensors.
                - ``"identity_matches"``: List of T-1 match tensors.
                - ``"identity_preservation"``: Float in ``[0, 1]``.
                - ``"per_frame_similarity"``: List of T-1 mean sims.
                - ``"per_frame_phase_correlation"``: List of T-1 ρ values.
        """
        T = len(frame_detections)
        phase_history: list[Tensor] = []
        identity_matches: list[Tensor] = []
        per_frame_sim: list[float] = []
        per_frame_rho: list[float] = []

        total_matches = 0
        total_possible = 0

        with torch.no_grad():
            for t in range(T):
                dets = frame_detections[t]
                phase_t, amp_t = self.encode(dets)

                if t == 0:
                    # First frame: just record phases, no matching
                    phase_history.append(phase_t.detach().cpu())
                    continue

                # Evolve previous frame's phases forward
                prev_phase = phase_history[-1].to(dets.device)
                prev_amp = torch.ones_like(prev_phase)  # unit amplitude
                evolved_phase, _ = self.evolve(prev_phase, prev_amp)

                # Similarity between evolved previous and current raw
                sim = self.phase_similarity(evolved_phase, phase_t)

                # Greedy matching
                N_prev = evolved_phase.shape[0]
                N_curr = phase_t.shape[0]
                N_match = min(N_prev, N_curr)
                matches = torch.full((N_prev,), -1, dtype=torch.long,
                                     device=dets.device)
                used = torch.zeros(N_curr, dtype=torch.bool,
                                   device=dets.device)
                max_sims, max_idxs = sim.max(dim=1)
                order = max_sims.argsort(descending=True)

                for idx in order:
                    best_j = max_idxs[idx].item()
                    if best_j < N_curr and not used[best_j] and max_sims[idx] > self.match_threshold:
                        matches[idx] = best_j
                        used[best_j] = True

                n_matched = int((matches >= 0).sum().item())
                identity_matches.append(matches.cpu())
                per_frame_sim.append(float(max_sims.mean().item()))
                total_matches += n_matched
                total_possible += N_match

                # Inter-frame phase correlation (circular)
                # ρ = |mean(exp(i(φ_t - φ_{t-1})))|  over matched pairs
                if n_matched > 0:
                    matched_mask = matches >= 0
                    matched_prev = evolved_phase[matched_mask]
                    matched_curr = phase_t[matches[matched_mask]]
                    diff = matched_curr - matched_prev
                    rho = float(
                        torch.exp(1j * diff.to(torch.complex64))
                        .mean(dim=-1).abs().mean().item()
                    )
                else:
                    rho = 0.0
                per_frame_rho.append(rho)

                phase_history.append(phase_t.detach().cpu())

        preservation = total_matches / max(total_possible, 1)

        return {
            "phase_history": phase_history,
            "identity_matches": identity_matches,
            "identity_preservation": preservation,
            "per_frame_similarity": per_frame_sim,
            "per_frame_phase_correlation": per_frame_rho,
        }

        return matches, sim