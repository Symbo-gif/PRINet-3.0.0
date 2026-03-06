"""Year 4 Q1 experiment utilities: ablation, extended CLEVR-N, FLOPs.

Provides infrastructure for paper-critical experiments (T.4, T.5, T.6):

- :class:`AblationConfig` — Configures HybridPRINetV2 ablation variants.
- :func:`create_ablation_model` — Builds ablation model from config.
- :func:`train_clevr_n_extended` — Multi-seed training with error bars.
- :func:`count_flops` — FLOPs estimation for forward pass.
- :func:`measure_wall_time` — Wall-time benchmarking utility.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from prinet.core.propagation import DiscreteDeltaThetaGamma

# =========================================================================
# T.5: Ablation Framework
# =========================================================================


@dataclass
class AblationConfig:
    """Configuration for HybridPRINetV2 ablation variants.

    Attributes:
        variant: One of ``"full"``, ``"attention_only"``,
            ``"oscillator_only"``, ``"shared_phase"``.
        n_input: Input dimension.
        n_classes: Number of classes.
        d_model: Model dimension.
        n_heads: Attention heads.
        n_layers: Number of layers.
        n_delta: Delta-band oscillators.
        n_theta: Theta-band oscillators.
        n_gamma: Gamma-band oscillators.
        n_discrete_steps: Dynamics steps per layer.
        coupling_strength: Coupling K.
        pac_depth: PAC modulation depth.
        dropout: Dropout rate.
    """

    variant: Literal["full", "attention_only", "oscillator_only", "shared_phase"] = (
        "full"
    )
    n_input: int = 256
    n_classes: int = 10
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_delta: int = 4
    n_theta: int = 8
    n_gamma: int = 32
    n_discrete_steps: int = 5
    coupling_strength: float = 2.0
    pac_depth: float = 0.3
    dropout: float = 0.1


class AblationHybridPRINetV2(nn.Module):
    """HybridPRINetV2 with configurable ablation of oscillatory components.

    Supports four ablation variants for isolating the contribution of
    oscillatory dynamics vs. attention in the hybrid architecture:

    - ``"full"``: Standard HybridPRINetV2 (no ablation).
    - ``"attention_only"``: Remove oscillatory dynamics; use standard
      positional encoding instead of phase modulation.
    - ``"oscillator_only"``: Remove attention mechanism; use only
      oscillatory phase binding with MLP mixing.
    - ``"shared_phase"``: All oscillators use the same frequency
      (removes frequency diversity, tests phase-locking).

    Args:
        config: Ablation configuration dataclass.

    Example:
        >>> cfg = AblationConfig(variant="attention_only")
        >>> model = AblationHybridPRINetV2(cfg)
        >>> x = torch.randn(8, 256)
        >>> log_probs = model(x)
    """

    _LOGIT_CLAMP = 20.0

    def __init__(self, config: AblationConfig) -> None:
        super().__init__()
        self.config = config
        self.variant = config.variant
        self.d_model = config.d_model
        self.n_layers = config.n_layers

        n_osc = config.n_delta + config.n_theta + config.n_gamma
        self.n_tokens = n_osc

        # Input projection
        self.input_proj = nn.Linear(config.n_input, n_osc * config.d_model)

        if config.variant != "oscillator_only":
            # Attention layers
            from prinet.nn.layers import OscillatoryAttention

            self.attn_layers: Optional[nn.ModuleList] = nn.ModuleList()
            self.norm1_layers: Optional[nn.ModuleList] = nn.ModuleList()
            for _ in range(config.n_layers):
                self.attn_layers.append(
                    OscillatoryAttention(
                        d_model=config.d_model,
                        n_heads=config.n_heads,
                        dropout=config.dropout,
                    )
                )
                self.norm1_layers.append(nn.LayerNorm(config.d_model))
        else:
            self.attn_layers = None
            self.norm1_layers = None

        # FFN layers (always present)
        self.ffn_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(config.d_model, config.d_model * 4),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_model * 4, config.d_model),
                    nn.Dropout(config.dropout),
                )
            )
            self.norm2_layers.append(nn.LayerNorm(config.d_model))

        if config.variant not in ("attention_only",):
            # Oscillatory dynamics
            from prinet.core.propagation import DiscreteDeltaThetaGamma

            if config.variant == "shared_phase":
                # Shared-phase: all oscillators at same frequency
                self.dynamics: Optional[DiscreteDeltaThetaGamma] = DiscreteDeltaThetaGamma(
                    n_delta=config.n_delta,
                    n_theta=config.n_theta,
                    n_gamma=config.n_gamma,
                    coupling_strength=config.coupling_strength,
                    pac_depth=config.pac_depth,
                )
                # We'll override frequencies after construction
                self._shared_phase = True
            else:
                self.dynamics = DiscreteDeltaThetaGamma(
                    n_delta=config.n_delta,
                    n_theta=config.n_theta,
                    n_gamma=config.n_gamma,
                    coupling_strength=config.coupling_strength,
                    pac_depth=config.pac_depth,
                )
                self._shared_phase = False

            self.phase_init: Optional[nn.Linear] = nn.Linear(config.n_input, n_osc * config.n_heads)
        else:
            self.dynamics = None
            self._shared_phase = False
            self.phase_init = None

        # Classification head
        self.pool_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with ablation-specific routing.

        Args:
            x: Input ``(B, D)`` or ``(D,)``.

        Returns:
            Log-probabilities ``(B, K)`` or ``(K,)``.
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        B = x.shape[0]

        # Embed as token sequence
        h = self.input_proj(x).view(B, self.n_tokens, self.d_model)

        # Phase computation
        phase_state: Optional[Tensor] = None
        if self.dynamics is not None and self.phase_init is not None:
            phase_raw = self.phase_init(x).view(B, self.n_tokens, self.config.n_heads)
            phase_state = phase_raw % (2.0 * math.pi)

            # For shared_phase: set all oscillator frequencies equal
            if self._shared_phase:
                phase_state = phase_state.mean(dim=1, keepdim=True).expand_as(
                    phase_state
                )

            amp_state = torch.ones(B, self.n_tokens, device=x.device, dtype=x.dtype)
            dyn_phase = phase_state.mean(dim=-1)

        for i in range(self.n_layers):
            # Oscillatory dynamics update (if not attention_only)
            token_phase: Optional[Tensor] = None
            if self.dynamics is not None and phase_state is not None:
                dyn_phase, amp_state = self.dynamics.integrate(
                    dyn_phase,
                    amp_state,
                    n_steps=self.config.n_discrete_steps,
                    dt=0.01,
                )
                token_phase = dyn_phase.unsqueeze(-1).expand(
                    B, self.n_tokens, self.config.n_heads
                )

            # Attention or MLP mixing
            if self.attn_layers is not None and self.norm1_layers is not None:
                h_norm = self.norm1_layers[i](h)
                if token_phase is not None:
                    h = h + self.attn_layers[i](h_norm, phase=token_phase)
                else:
                    # attention_only: use zero phase
                    zero_phase = torch.zeros(
                        B,
                        self.n_tokens,
                        self.config.n_heads,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    h = h + self.attn_layers[i](h_norm, phase=zero_phase)
            else:
                # oscillator_only: simple token mixing via mean pool
                h_mixed = h.mean(dim=1, keepdim=True).expand_as(h)
                h = h + 0.1 * h_mixed

            # FFN
            h_norm = self.norm2_layers[i](h)
            h = h + self.ffn_layers[i](h_norm)

        # Pool and classify
        pooled = self.pool_norm(h.mean(dim=1))
        logits = self.classifier(pooled)
        logits = torch.clamp(logits, -self._LOGIT_CLAMP, self._LOGIT_CLAMP)
        log_probs = F.log_softmax(logits, dim=-1)

        if was_1d:
            return log_probs.squeeze(0)
        return log_probs


def create_ablation_model(
    variant: str = "full",
    n_input: int = 256,
    n_classes: int = 10,
    **kwargs: Any,
) -> AblationHybridPRINetV2:
    """Create an ablation model variant.

    Args:
        variant: One of ``"full"``, ``"attention_only"``,
            ``"oscillator_only"``, ``"shared_phase"``.
        n_input: Input feature dimension.
        n_classes: Number of output classes.
        **kwargs: Passed through to :class:`AblationConfig`.

    Returns:
        Configured :class:`AblationHybridPRINetV2`.
    """
    config = AblationConfig(
        variant=variant,  # type: ignore[arg-type]
        n_input=n_input,
        n_classes=n_classes,
        **kwargs,
    )
    return AblationHybridPRINetV2(config)


# =========================================================================
# T.4: Extended CLEVR-N Multi-Seed Training Infrastructure
# =========================================================================


@dataclass
class ExtendedTrainingResult:
    """Result from multi-seed extended CLEVR-N training.

    Attributes:
        model_name: Name of the model architecture.
        n_objects: Number of objects in the CLEVR-N task.
        n_seeds: Number of random seeds.
        n_epochs: Number of epochs per seed.
        accuracies: Per-seed final accuracies.
        mean_accuracy: Mean accuracy across seeds.
        std_accuracy: Standard deviation.
        losses: Per-seed final losses.
        mean_loss: Mean final loss.
        std_loss: Std of final loss.
        p_value: Two-sample t-test p-value (if comparison provided).
        wall_times: Per-seed training wall time.
    """

    model_name: str = ""
    n_objects: int = 0
    n_seeds: int = 0
    n_epochs: int = 0
    accuracies: list[float] = field(default_factory=list)
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    losses: list[float] = field(default_factory=list)
    mean_loss: float = 0.0
    std_loss: float = 0.0
    p_value: Optional[float] = None
    wall_times: list[float] = field(default_factory=list)


def train_clevr_n_single_seed(
    model: nn.Module,
    n_objects: int = 6,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[float, float, float]:
    """Train a model on synthetic CLEVR-N for one seed.

    Generates synthetic scene + query data and trains the model to
    classify relational queries. Returns final accuracy, loss, and
    training time.

    Args:
        model: Model accepting (scene, query) → log_probs.
        n_objects: Number of objects in the scene.
        n_epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        seed: Random seed.
        device: Device string.

    Returns:
        Tuple of (accuracy, loss, wall_time_s).
    """
    torch.manual_seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Generate synthetic CLEVR-N data
    n_train = 256
    n_test = 64
    scene_dim = 16
    query_dim = 60

    # Synthetic data: random features, random binary labels
    train_scenes = torch.randn(n_train, n_objects, scene_dim, device=device)
    train_queries = torch.randn(n_train, query_dim, device=device)
    train_labels = torch.randint(0, 2, (n_train,), device=device)

    test_scenes = torch.randn(n_test, n_objects, scene_dim, device=device)
    test_queries = torch.randn(n_test, query_dim, device=device)
    test_labels = torch.randint(0, 2, (n_test,), device=device)

    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            scenes = train_scenes[idx]
            queries = train_queries[idx]
            labels = train_labels[idx]

            optimizer.zero_grad()
            log_probs = model(scenes, queries)
            loss = F.nll_loss(log_probs, labels)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

    wall_time = time.perf_counter() - t0

    # Evaluate
    model.eval()
    with torch.no_grad():
        log_probs = model(test_scenes, test_queries)
        preds = log_probs.argmax(dim=-1)
        accuracy = float((preds == test_labels).float().mean().item())
        test_loss = float(F.nll_loss(log_probs, test_labels).item())

    return accuracy, test_loss, wall_time


def train_clevr_n_extended(
    model_factory: Any,
    model_name: str = "model",
    n_objects: int = 6,
    n_seeds: int = 5,
    n_epochs: int = 100,
    device: str = "cpu",
    **train_kwargs: Any,
) -> ExtendedTrainingResult:
    """Train a model on CLEVR-N across multiple seeds with statistics.

    Args:
        model_factory: Callable that returns a fresh model instance.
        model_name: Name for logging.
        n_objects: Number of CLEVR-N objects.
        n_seeds: Number of random seeds.
        n_epochs: Epochs per seed.
        device: Device string.
        **train_kwargs: Passed to :func:`train_clevr_n_single_seed`.

    Returns:
        :class:`ExtendedTrainingResult` with mean, std, and per-seed results.
    """
    accs: list[float] = []
    losses: list[float] = []
    times: list[float] = []

    for s in range(n_seeds):
        model = model_factory()
        acc, loss, wt = train_clevr_n_single_seed(
            model,
            n_objects=n_objects,
            n_epochs=n_epochs,
            seed=42 + s,
            device=device,
            **train_kwargs,
        )
        accs.append(acc)
        losses.append(loss)
        times.append(wt)

    mean_acc = sum(accs) / len(accs)
    std_acc = (sum((a - mean_acc) ** 2 for a in accs) / max(len(accs) - 1, 1)) ** 0.5
    mean_loss = sum(losses) / len(losses)
    std_loss = (
        sum((l - mean_loss) ** 2 for l in losses) / max(len(losses) - 1, 1)
    ) ** 0.5

    return ExtendedTrainingResult(
        model_name=model_name,
        n_objects=n_objects,
        n_seeds=n_seeds,
        n_epochs=n_epochs,
        accuracies=accs,
        mean_accuracy=mean_acc,
        std_accuracy=std_acc,
        losses=losses,
        mean_loss=mean_loss,
        std_loss=std_loss,
        wall_times=times,
    )


def compute_p_value(
    accs_a: list[float],
    accs_b: list[float],
) -> float:
    """Compute two-sample Welch's t-test p-value.

    Args:
        accs_a: Accuracies from model A.
        accs_b: Accuracies from model B.

    Returns:
        Two-tailed p-value.
    """
    from scipy.stats import ttest_ind

    result = ttest_ind(accs_a, accs_b, equal_var=False)
    return float(result.pvalue)


# =========================================================================
# T.6: FLOPs Counting & Efficiency Comparison
# =========================================================================


def count_flops(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cpu",
) -> dict[str, Any]:
    """Estimate FLOPs for a model's forward pass.

    Uses a simple parameter-based estimation: for each linear layer,
    FLOPs ≈ 2 * in_features * out_features * batch_size. For conv
    layers, FLOPs ≈ 2 * C_in * C_out * K² * H_out * W_out.

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape (e.g., ``(8, 256)``).
        device: Device for computation.

    Returns:
        Dict with ``"total_flops"``, ``"total_params"``,
        ``"layer_flops"`` (list of per-layer dicts).
    """
    model = model.to(device)
    total_flops = 0
    layer_details: list[dict[str, Any]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # FLOPs = 2 * in_features * out_features (multiply-add)
            flops = 2 * module.in_features * module.out_features
            if module.bias is not None:
                flops += module.out_features
            total_flops += flops
            layer_details.append(
                {
                    "name": name,
                    "type": "Linear",
                    "flops": flops,
                    "params": module.in_features * module.out_features
                    + (module.out_features if module.bias is not None else 0),
                }
            )
        elif isinstance(module, nn.Conv2d):
            flops = (
                2
                * module.in_channels
                * module.out_channels
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
            total_flops += flops
            layer_details.append(
                {
                    "name": name,
                    "type": "Conv2d",
                    "flops": flops,
                    "params": sum(p.numel() for p in module.parameters()),
                }
            )
        elif isinstance(module, nn.GRUCell):
            # GRUCell: 3 gates × 2 × (input_size + hidden_size) × hidden_size
            flops = (
                3 * 2 * (module.input_size + module.hidden_size) * module.hidden_size
            )
            total_flops += flops
            layer_details.append(
                {
                    "name": name,
                    "type": "GRUCell",
                    "flops": flops,
                    "params": sum(p.numel() for p in module.parameters()),
                }
            )

    # Scale by batch size
    batch_size = input_shape[0] if len(input_shape) > 1 else 1
    total_flops *= batch_size

    total_params = sum(p.numel() for p in model.parameters())

    return {
        "total_flops": total_flops,
        "total_params": total_params,
        "layer_flops": layer_details,
    }


def measure_wall_time(
    model: nn.Module,
    input_tensor: "Tensor | tuple[Tensor, ...]",
    n_warmup: int = 5,
    n_runs: int = 20,
) -> dict[str, float]:
    """Measure forward-pass wall time with warmup.

    Args:
        model: PyTorch model.
        input_tensor: Input tensor or tuple of input tensors.
        n_warmup: Number of warmup runs.
        n_runs: Number of timed runs.

    Returns:
        Dict with ``"mean_ms"``, ``"std_ms"``, ``"min_ms"``,
        ``"max_ms"``.
    """
    model.eval()
    if isinstance(input_tensor, (tuple, list)):
        device = input_tensor[0].device
        _call = lambda: model(*input_tensor)
    else:
        device = input_tensor.device
        _call = lambda: model(input_tensor)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _call()  # type: ignore[no-untyped-call]

    # Timed runs
    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _call()  # type: ignore[no-untyped-call]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed)

    mean_t = sum(times) / len(times)
    std_t = (sum((t - mean_t) ** 2 for t in times) / max(len(times) - 1, 1)) ** 0.5

    return {
        "mean_ms": mean_t,
        "std_ms": std_t,
        "min_ms": min(times),
        "max_ms": max(times),
    }


# =========================================================================
# Q1.2: Statistical Utilities for Rigorous Benchmarking
# =========================================================================


def chimera_initial_condition(N: int, seed: int = 0) -> Tensor:
    """Generate single-humped initial phase for chimera emergence.

    Following Abrams & Strogatz (2006), a localised bump initial
    condition dramatically increases the probability of chimera
    formation. The formula is:

    .. math::

        \\phi_i = 6 \\exp\\bigl(-30 (x_i - 0.5)^2\\bigr) \\cdot r_i

    where :math:`x_i \\in [0, 1)` is the normalised position and
    :math:`r_i \\sim U(-0.5, 0.5)` is a small random perturbation.

    Args:
        N: Number of oscillators.
        seed: Random seed for the perturbation.

    Returns:
        Phase tensor ``(N,)`` suitable for ``OscilloSim.run(initial_phase=...)``.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.linspace(0, 1, N)
    bump = 6.0 * torch.exp(-30.0 * (x - 0.5) ** 2)
    noise = torch.rand(N, generator=gen) - 0.5
    return bump * noise


def gaussian_bump_ic(
    N: int,
    A0: float = math.pi,
    sigma_ratio: float = 1 / 6,
    phi0: float = 0.0,
    noise_amp: float = 0.01,
    seed: int = 0,
) -> Tensor:
    """Smooth Gaussian-bump initial condition for chimera states.

    Generates an initial phase profile:

    .. math::

        \\theta_i = \\phi_0 + A_0 \\exp\\!\\left(
            -\\frac{(i - i_0)^2}{2\\sigma_0^2}\\right)
            + \\varepsilon_i

    where :math:`i_0 = N/2`, :math:`\\sigma_0 = \\sigma_{\\text{ratio}} N`,
    and :math:`\\varepsilon_i \\sim U(-a, a)` with :math:`a` =
    ``noise_amp``.  This profile seeds chimera formation by creating
    a localised perturbation in an otherwise nearly uniform population.

    Args:
        N: Number of oscillators.
        A0: Amplitude of the Gaussian bump (radians).
        sigma_ratio: σ/N ratio (default 1/6 → wide bump).
        phi0: Baseline phase offset.
        noise_amp: Uniform noise amplitude.
        seed: Random seed.

    Returns:
        Phase tensor ``(N,)`` in radians.
    """
    gen = torch.Generator().manual_seed(seed)
    i = torch.arange(N, dtype=torch.float32)
    i0 = N / 2.0
    sigma = sigma_ratio * N
    bump = A0 * torch.exp(-((i - i0) ** 2) / (2.0 * sigma**2))
    noise = (torch.rand(N, generator=gen) - 0.5) * 2.0 * noise_amp
    return (phi0 + bump + noise) % (2.0 * math.pi)


def half_sync_half_random_ic(
    N: int,
    sync_phase: float = 0.0,
    noise_amp: float = 0.01,
    seed: int = 0,
) -> Tensor:
    """Half-synchronised, half-random initial condition.

    The first ``N/2`` oscillators are set near ``sync_phase`` (with
    tiny noise), while the remaining oscillators are uniformly
    distributed over :math:`[0, 2\\pi)`.  This setup directly seeds
    chimera states with a clean co-existing coherent/incoherent split.

    Args:
        N: Number of oscillators.
        sync_phase: Phase of the synchronised half (radians).
        noise_amp: Noise amplitude for the coherent half.
        seed: Random seed.

    Returns:
        Phase tensor ``(N,)`` in radians.
    """
    gen = torch.Generator().manual_seed(seed)
    half = N // 2
    coherent = sync_phase + (torch.rand(half, generator=gen) - 0.5) * 2.0 * noise_amp
    incoherent = torch.rand(N - half, generator=gen) * 2.0 * math.pi
    return torch.cat([coherent, incoherent]) % (2.0 * math.pi)


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for the mean.

    Uses the percentile method to construct a ``(1 - alpha)`` CI.

    Args:
        values: Observed values.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 → 95 % CI).
        seed: Random seed.

    Returns:
        Dict with ``"mean"``, ``"ci_lower"``, ``"ci_upper"``,
        ``"ci_width"``, ``"se"``.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()

    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {
        "mean": float(arr.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_width": hi - lo,
        "se": float(means.std()),
    }


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Compute Cohen's d effect size (pooled standard deviation).

    .. math::

        d = \\frac{\\bar{x}_A - \\bar{x}_B}{s_p}

    where :math:`s_p = \\sqrt{\\frac{(n_A-1)s_A^2 + (n_B-1)s_B^2}{n_A+n_B-2}}`.

    Args:
        group_a: Observations from condition A.
        group_b: Observations from condition B.

    Returns:
        Cohen's d (positive means A > B).
    """
    import numpy as np

    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def welch_t_test(
    group_a: list[float],
    group_b: list[float],
) -> dict[str, float]:
    """Welch's t-test with effect size and Bonferroni-ready p-value.

    Args:
        group_a: Observations from condition A.
        group_b: Observations from condition B.

    Returns:
        Dict with ``"t_stat"``, ``"p_value"``, ``"cohens_d"``,
        ``"mean_diff"``.
    """
    from scipy.stats import ttest_ind

    result = ttest_ind(group_a, group_b, equal_var=False)
    d = cohens_d(group_a, group_b)
    import numpy as np

    return {
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "cohens_d": d,
        "mean_diff": float(np.mean(group_a) - np.mean(group_b)),
    }


def spatial_correlation(
    r_local: Tensor,
    max_lag: int = 50,
) -> list[float]:
    """Compute spatial autocorrelation of local order parameter.

    For chimera states, the spatial autocorrelation decays in the
    incoherent region but remains high in the coherent region, producing
    a characteristic pattern with a secondary peak at the inter-domain
    distance.

    Args:
        r_local: Local order parameter ``(N,)``.
        max_lag: Maximum spatial lag to compute.

    Returns:
        List of autocorrelation values for lags 0..max_lag.
    """
    v = r_local.float().cpu()
    v = v - v.mean()
    var = (v**2).mean()
    if var < 1e-12:
        return [1.0] + [0.0] * max_lag
    result = []
    N = v.numel()
    max_lag = min(max_lag, N // 2)
    for lag in range(max_lag + 1):
        shifted = torch.roll(v, lag)
        corr = float((v * shifted).mean() / var)
        result.append(corr)
    return result


def seed_stability_analysis(
    per_seed_results: list[dict[str, Any]],
    metric_key: str,
) -> dict[str, float]:
    """Analyse stability of a metric across random seeds.

    Args:
        per_seed_results: List of dicts, each with ``metric_key``.
        metric_key: Key to extract from each result dict.

    Returns:
        Dict with ``"mean"``, ``"std"``, ``"cv"`` (coefficient of
        variation), ``"range"``, ``"n_seeds"``.
    """
    vals = [r[metric_key] for r in per_seed_results]
    mean_v = sum(vals) / len(vals)
    std_v = (sum((v - mean_v) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return {
        "mean": mean_v,
        "std": std_v,
        "cv": std_v / abs(mean_v) if abs(mean_v) > 1e-12 else 0.0,
        "range": max(vals) - min(vals),
        "n_seeds": len(vals),
    }


# =========================================================================
# Y4 Q1.4: Temporal Advantage Metrics
# =========================================================================


def phase_slip_rate(
    phase_trajectory: Tensor,
    threshold: float = math.pi * 0.8,
) -> dict[str, Any]:
    """Compute phase-slip rate from a phase trajectory.

    A *phase slip* occurs when the circular (geodesic) distance between
    consecutive phases exceeds ``threshold`` (default 0.8π ≈ 144°).
    Because phases live on S¹, the maximum geodesic distance is π;
    the default 0.8π catches near-antipodal jumps while allowing
    moderate phase evolution.

    The metric is adapted from clinical neurophysiology (epilepsy
    seizure analysis) and counts abrupt near-π discontinuities in
    the circular phase signal.

    Args:
        phase_trajectory: Phase history ``(T, N)`` where *T* is the
            number of time steps and *N* is the number of oscillators.
            Phases should be in radians.
        threshold: Circular-distance threshold (default 0.8π) above
            which a transition is counted as a slip.  Must be < π
            for meaningful results since wrapped diffs lie in [-π, π].

    Returns:
        Dict with:
            - ``"total_slips"``: Total slip count across all oscillators.
            - ``"slips_per_step"``: Mean slips per time step.
            - ``"slips_per_oscillator"``: Mean slips per oscillator.
            - ``"slip_fraction"``: Fraction of (step, oscillator) pairs
              that experienced a slip.
            - ``"per_oscillator_slips"``: List of per-oscillator slip
              counts.

    Example:
        >>> trajectory = torch.rand(50, 10) * 2 * math.pi
        >>> result = phase_slip_rate(trajectory)
        >>> assert 0.0 <= result["slip_fraction"] <= 1.0
    """
    if phase_trajectory.dim() != 2:
        raise ValueError(
            f"Expected 2-D (T, N) trajectory, got shape " f"{phase_trajectory.shape}."
        )
    T, N = phase_trajectory.shape
    if T < 2:
        return {
            "total_slips": 0,
            "slips_per_step": 0.0,
            "slips_per_oscillator": 0.0,
            "slip_fraction": 0.0,
            "per_oscillator_slips": [0] * N,
        }

    # Compute wrapped phase differences (circular)
    diff = phase_trajectory[1:] - phase_trajectory[:-1]  # (T-1, N)
    # Wrap to [-π, π]
    diff = (diff + math.pi) % (2.0 * math.pi) - math.pi
    slips = (diff.abs() > threshold).long()  # (T-1, N)

    per_osc = slips.sum(dim=0).tolist()  # list[int], length N
    total = int(slips.sum().item())
    n_transitions = (T - 1) * N

    return {
        "total_slips": total,
        "slips_per_step": total / (T - 1),
        "slips_per_oscillator": total / N,
        "slip_fraction": total / n_transitions,
        "per_oscillator_slips": per_osc,
    }


def binding_persistence(
    matches_history: list[Tensor],
    n_objects: int,
) -> dict[str, Any]:
    """Measure how persistently objects maintain identity across frames.

    Computes the fraction of frames in which each object is
    *continuously* tracked (matched), and reports aggregate statistics.

    A persistent binding means the same object index is matched to
    *some* target (≥ 0) in every frame; a break is when a match drops
    to −1 (unmatched).

    Args:
        matches_history: List of T-1 match tensors, each ``(N,)``
            where values ≥ 0 indicate a valid match and −1 means
            unmatched.
        n_objects: Number of objects (should equal the length of each
            match tensor).

    Returns:
        Dict with:
            - ``"mean_persistence"``: Mean persistence ratio (0–1) over
              all object slots.
            - ``"min_persistence"``: Worst-case slot persistence.
            - ``"per_object_persistence"``: List of per-slot values.
            - ``"n_frames"``: Number of frame transitions.
    """
    if not matches_history:
        return {
            "mean_persistence": 0.0,
            "min_persistence": 0.0,
            "per_object_persistence": [],
            "n_frames": 0,
        }

    T = len(matches_history)
    per_obj: list[float] = []
    for obj_idx in range(n_objects):
        matched_count = sum(1 for m in matches_history if m[obj_idx].item() >= 0)
        per_obj.append(matched_count / T)

    return {
        "mean_persistence": sum(per_obj) / len(per_obj),
        "min_persistence": min(per_obj),
        "per_object_persistence": per_obj,
        "n_frames": T,
    }


def coherence_decay_rate(
    coherence_series: list[float],
) -> dict[str, float]:
    r"""Fit an exponential decay to a coherence time series.

    Models coherence as :math:`C(t) = C_0 \exp(-\lambda t)` and
    reports the decay rate :math:`\lambda`.  Lower :math:`\lambda`
    indicates more persistent coherence — a key temporal advantage
    metric.

    If coherence is non-monotonic a simple linear fit to
    :math:`\ln C(t)` is performed.

    Args:
        coherence_series: List of coherence values (one per frame
            transition).  Values should be in ``(0, 1]``.

    Returns:
        Dict with:
            - ``"decay_rate"``: Fitted λ (higher = faster decay).
            - ``"half_life"``: Frames until coherence halves (ln2/λ).
            - ``"initial_coherence"``: Fitted C₀.
            - ``"r_squared"``: Goodness of fit.
    """
    import numpy as np

    c = np.asarray(coherence_series, dtype=np.float64)
    # Clamp to avoid log(0)
    c = np.clip(c, 1e-10, None)
    n = len(c)
    if n < 2:
        return {
            "decay_rate": 0.0,
            "half_life": float("inf"),
            "initial_coherence": float(c[0]) if n else 0.0,
            "r_squared": 0.0,
        }

    t = np.arange(n, dtype=np.float64)
    ln_c = np.log(c)

    # Linear regression: ln(C) = ln(C0) - λ*t
    A = np.vstack([t, np.ones(n)]).T
    result = np.linalg.lstsq(A, ln_c, rcond=None)
    slope, intercept = result[0]
    lam = -slope  # decay rate
    c0 = math.exp(intercept)

    # R²
    predicted = slope * t + intercept
    ss_res = float(np.sum((ln_c - predicted) ** 2))
    ss_tot = float(np.sum((ln_c - ln_c.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    half_life = math.log(2) / max(abs(lam), 1e-12) if lam > 0 else float("inf")

    return {
        "decay_rate": float(lam),
        "half_life": float(half_life),
        "initial_coherence": float(c0),
        "r_squared": float(r2),
    }


def rebinding_speed(
    matches_before: list[Tensor],
    matches_after: list[Tensor],
    n_objects: int,
) -> dict[str, Any]:
    """Measure how quickly identity bindings recover after perturbation.

    Compares match quality before a perturbation (last *k* frames of
    ``matches_before``) to the recovery trajectory in
    ``matches_after`` and counts how many frames are needed for the
    matching rate to return to the pre-perturbation level.

    Args:
        matches_before: Matches from the pre-perturbation segment.
        matches_after: Matches from the post-perturbation segment.
        n_objects: Number of object slots.

    Returns:
        Dict with:
            - ``"pre_match_rate"``: Mean match rate before perturbation.
            - ``"recovery_frames"``: Frames to recover 90 % of
              pre-match rate, or ``-1`` if never recovered.
            - ``"post_match_rates"``: List of per-frame match rates
              in the recovery window.
    """
    if not matches_before:
        pre_rate = 1.0
    else:
        rates_before = [(m >= 0).float().mean().item() for m in matches_before]
        pre_rate = sum(rates_before) / len(rates_before)

    target = 0.9 * pre_rate
    post_rates: list[float] = []
    recovery_frame = -1

    for i, m in enumerate(matches_after):
        rate = (m >= 0).float().mean().item()
        post_rates.append(rate)
        if rate >= target and recovery_frame < 0:
            recovery_frame = i

    return {
        "pre_match_rate": pre_rate,
        "recovery_frames": recovery_frame,
        "post_match_rates": post_rates,
    }


def cross_frequency_coupling(
    phases_low: Tensor,
    phases_high: Tensor,
) -> dict[str, Any]:
    r"""Compute phase-amplitude coupling (PAC) between frequency bands.

    Measures hierarchical phase coupling between low-frequency (e.g.
    delta/theta) and high-frequency (e.g. gamma) oscillators.  High
    PAC during successful tracking implies oscillatory binding
    organises high-frequency dynamics via low-frequency coordination —
    a hallmark of binding-by-synchrony theory.

    .. math::

        \text{PAC} = \left| \frac{1}{N}
            \sum_{i} \sin(\phi_{\text{low},i} - \phi_{\text{high},i})
        \right|

    Args:
        phases_low: Low-frequency phases ``(N,)`` or ``(T, N)``.
        phases_high: High-frequency phases ``(N,)`` or ``(T, N)``.
            Must have the same shape as ``phases_low``.

    Returns:
        Dict with:
            - ``"pac"``: Mean PAC value in ``[0, 1]``.
            - ``"pac_per_step"``: List of per-step PAC values (if 2-D).
    """
    if phases_low.shape != phases_high.shape:
        raise ValueError(
            f"Shape mismatch: phases_low {phases_low.shape} vs "
            f"phases_high {phases_high.shape}."
        )

    diff = phases_low - phases_high
    if phases_low.dim() == 1:
        pac_val = float(torch.sin(diff).mean().abs().item())
        return {"pac": pac_val, "pac_per_step": [pac_val]}

    # 2-D: (T, N)
    per_step = [
        float(torch.sin(diff[t]).mean().abs().item()) for t in range(diff.shape[0])
    ]
    return {
        "pac": sum(per_step) / len(per_step),
        "pac_per_step": per_step,
    }


def temporal_advantage_report(
    phase_tracker_result: dict[str, Any],
    slot_attention_result: dict[str, Any],
    n_seeds: int = 1,
) -> dict[str, Any]:
    """Compile a head-to-head temporal advantage report.

    Compares the outputs of
    :meth:`~prinet.nn.hybrid.PhaseTracker.track_sequence` and
    :meth:`~prinet.nn.slot_attention.TemporalSlotAttentionMOT.track_sequence`
    and computes advantage metrics.

    Args:
        phase_tracker_result: Dict from PhaseTracker.track_sequence.
        slot_attention_result: Dict from TemporalSlotAttentionMOT.track_sequence.
        n_seeds: Number of seeds that produced these results (for
            documentation only).

    Returns:
        Dict with:
            - ``"ip_phase"``: PhaseTracker identity preservation.
            - ``"ip_slot"``: SlotAttention identity preservation.
            - ``"ip_advantage"``: ip_phase − ip_slot (positive = phase wins).
            - ``"mean_sim_phase"``: Mean per-frame similarity (PhaseTracker).
            - ``"mean_sim_slot"``: Mean per-frame similarity (SlotAttention).
            - ``"mean_rho_phase"``: Mean phase correlation (PhaseTracker only).
            - ``"n_seeds"``: Seed count.
    """
    ip_phase = phase_tracker_result["identity_preservation"]
    ip_slot = slot_attention_result["identity_preservation"]

    sim_phase = phase_tracker_result.get("per_frame_similarity", [])
    sim_slot = slot_attention_result.get("per_frame_similarity", [])
    rho_phase = phase_tracker_result.get("per_frame_phase_correlation", [])

    return {
        "ip_phase": ip_phase,
        "ip_slot": ip_slot,
        "ip_advantage": ip_phase - ip_slot,
        "mean_sim_phase": sum(sim_phase) / max(len(sim_phase), 1),
        "mean_sim_slot": sum(sim_slot) / max(len(sim_slot), 1),
        "mean_rho_phase": sum(rho_phase) / max(len(rho_phase), 1),
        "n_seeds": n_seeds,
    }


# =========================================================================
# Y4 Q1.5: Session-length metrics
# =========================================================================


def order_parameter_series(
    phase_trajectory: Tensor,
) -> dict[str, Any]:
    """Compute the Kuramoto order parameter r(t) across a phase trajectory.

    The complex Kuramoto order parameter is:

    .. math::

        r(t) e^{i\\psi(t)} = \\frac{1}{N}\\sum_{j=1}^{N} e^{i\\theta_j(t)}

    Args:
        phase_trajectory: Phase history ``(T, N)`` or ``(T, N, K)``
            where *T* = time steps, *N* = oscillators (or objects), and
            *K* = oscillator bands (flattened internally).

    Returns:
        Dict with:
            - ``"r_series"``: List of *T* order parameter magnitudes.
            - ``"mean_r"``: Time-averaged r.
            - ``"std_r"``: Standard deviation of r.
            - ``"final_r"``: r at the last time step.
    """
    if phase_trajectory.dim() == 3:
        T, N, K = phase_trajectory.shape
        phase_trajectory = phase_trajectory.reshape(T, N * K)
    T, N = phase_trajectory.shape
    z = torch.exp(1j * phase_trajectory.to(torch.complex64))  # (T, N)
    r_complex = z.mean(dim=1)  # (T,)
    r_mag = r_complex.abs().float()  # (T,)
    r_list = r_mag.tolist()
    return {
        "r_series": r_list,
        "mean_r": float(r_mag.mean().item()),
        "std_r": float(r_mag.std().item()) if T > 1 else 0.0,
        "final_r": r_list[-1] if r_list else 0.0,
    }


def windowed_order_parameter_variance(
    r_series: list[float],
    window_size: int = 10,
) -> dict[str, Any]:
    """Compute windowed variance of the order parameter r(t).

    Divides *r_series* into non-overlapping windows and computes the
    standard deviation within each window.  Increasing windowed
    variance over time indicates instability.

    Args:
        r_series: Sequence of r(t) values.
        window_size: Number of samples per window.

    Returns:
        Dict with:
            - ``"window_stds"``: List of per-window standard deviations.
            - ``"trend_slope"``: Linear regression slope of window stds
              (positive = growing instability).
            - ``"mean_window_std"``: Mean of windowed stds.
    """
    import numpy as np

    n = len(r_series)
    n_windows = n // max(window_size, 1)
    if n_windows < 2:
        return {
            "window_stds": [float(np.std(r_series))] if r_series else [0.0],
            "trend_slope": 0.0,
            "mean_window_std": float(np.std(r_series)) if r_series else 0.0,
        }
    stds: list[float] = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        w = r_series[start:end]
        stds.append(float(np.std(w)))

    x = np.arange(len(stds), dtype=float)
    y = np.array(stds, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(stds) >= 2 else 0.0

    return {
        "window_stds": stds,
        "trend_slope": slope,
        "mean_window_std": float(np.mean(stds)),
    }


def phase_locking_value(
    phase_a: Tensor,
    phase_b: Tensor,
) -> dict[str, Any]:
    """Compute phase locking value (PLV) between two phase signals.

    .. math::

        PLV = \\left| \\frac{1}{T} \\sum_{t=1}^{T}
        e^{i(\\theta_a(t) - \\theta_b(t))} \\right|

    Args:
        phase_a: Phase signal ``(T,)`` or ``(T, N)``.
        phase_b: Phase signal, same shape as *phase_a*.

    Returns:
        Dict with:
            - ``"plv"``: Mean PLV across all oscillator pairs.
            - ``"plv_per_pair"``: Per-pair PLV if N > 1.
    """
    diff = phase_a - phase_b
    z = torch.exp(1j * diff.to(torch.complex64))
    if z.dim() == 1:
        plv_val = float(z.mean().abs().item())
        return {"plv": plv_val, "plv_per_pair": [plv_val]}
    # z is (T, N)
    plv_per_pair = z.mean(dim=0).abs().float().tolist()
    return {
        "plv": float(sum(plv_per_pair) / len(plv_per_pair)),
        "plv_per_pair": plv_per_pair,
    }


def instantaneous_frequency_spread(
    phase_trajectory: Tensor,
    dt: float = 1.0,
) -> dict[str, Any]:
    """Compute instantaneous frequency spread across oscillators.

    Differentiates the phase signal to get instantaneous frequency
    ω_i(t) = dθ_i/dt, then computes the spread (std) across oscillators
    at each time step.

    Args:
        phase_trajectory: ``(T, N)`` phase values.
        dt: Time step for numerical differentiation.

    Returns:
        Dict with:
            - ``"freq_spread_series"``: Per-step std of instantaneous freq.
            - ``"mean_spread"``: Time-averaged spread.
            - ``"trend_slope"``: Slope of spread over time (positive = diverging).
    """
    import numpy as np

    if phase_trajectory.dim() != 2 or phase_trajectory.shape[0] < 2:
        return {
            "freq_spread_series": [],
            "mean_spread": 0.0,
            "trend_slope": 0.0,
        }
    # Wrapped diff → instantaneous frequency
    diff = phase_trajectory[1:] - phase_trajectory[:-1]
    diff = (diff + math.pi) % (2.0 * math.pi) - math.pi
    omega = diff / dt  # (T-1, N)

    spread = omega.std(dim=1).tolist()  # std across oscillators per step

    x = np.arange(len(spread), dtype=float)
    y = np.array(spread, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(spread) >= 2 else 0.0

    return {
        "freq_spread_series": spread,
        "mean_spread": float(np.mean(spread)) if spread else 0.0,
        "trend_slope": slope,
    }


def cumulative_phase_slip_curve(
    phase_trajectory: Tensor,
    threshold: float = math.pi * 0.8,
) -> dict[str, Any]:
    """Compute cumulative phase-slip curve over session duration.

    Returns the running total of phase slips at each time step.
    A flat curve = stable binding.  An accelerating curve = degrading.

    Args:
        phase_trajectory: ``(T, N)`` phase values.
        threshold: Slip detection threshold.

    Returns:
        Dict with:
            - ``"cumulative_slips"``: List of cumulative slip counts.
            - ``"total_slips"``: Final total.
            - ``"acceleration"``: Quadratic coefficient of cumulative
              curve (positive = accelerating degradation).
    """
    import numpy as np

    if phase_trajectory.dim() != 2 or phase_trajectory.shape[0] < 2:
        return {"cumulative_slips": [], "total_slips": 0, "acceleration": 0.0}

    diff = phase_trajectory[1:] - phase_trajectory[:-1]
    diff = (diff + math.pi) % (2.0 * math.pi) - math.pi
    slip_per_step = (diff.abs() > threshold).sum(dim=1).tolist()  # per step

    cum = []
    running = 0
    for s in slip_per_step:
        running += s
        cum.append(running)

    total = cum[-1] if cum else 0
    accel = 0.0
    if len(cum) >= 3:
        x = np.arange(len(cum), dtype=float)
        y = np.array(cum, dtype=float)
        coeffs = np.polyfit(x, y, 2)
        accel = float(coeffs[0])

    return {
        "cumulative_slips": cum,
        "total_slips": total,
        "acceleration": accel,
    }


def throughput_series(
    wall_times_per_interval: list[float],
    frames_per_interval: int,
) -> dict[str, Any]:
    """Compute throughput stability metrics from interval timings.

    Args:
        wall_times_per_interval: Wall-clock seconds for each fixed-frame
            interval (e.g., every 100 frames).
        frames_per_interval: Number of frames processed per interval.

    Returns:
        Dict with:
            - ``"fps_series"``: Frames-per-second for each interval.
            - ``"mean_fps"``: Average FPS.
            - ``"std_fps"``: FPS standard deviation.
            - ``"degradation_pct"``: Percent throughput change from
              first to last interval (negative = degradation).
    """
    import numpy as np

    fps = [frames_per_interval / max(w, 1e-9) for w in wall_times_per_interval]
    if len(fps) < 2:
        return {
            "fps_series": fps,
            "mean_fps": fps[0] if fps else 0.0,
            "std_fps": 0.0,
            "degradation_pct": 0.0,
        }
    first_fps = fps[0]
    last_fps = fps[-1]
    degrad = ((last_fps - first_fps) / max(abs(first_fps), 1e-9)) * 100.0

    return {
        "fps_series": fps,
        "mean_fps": float(np.mean(fps)),
        "std_fps": float(np.std(fps)),
        "degradation_pct": degrad,
    }


def memory_growth_profile(
    memory_samples_mb: list[float],
    interval_seconds: float,
) -> dict[str, Any]:
    """Characterize memory growth over a session.

    Args:
        memory_samples_mb: Memory usage (MB) sampled at regular intervals.
        interval_seconds: Seconds between samples.

    Returns:
        Dict with:
            - ``"initial_mb"``: First sample.
            - ``"peak_mb"``: Maximum observed.
            - ``"final_mb"``: Last sample.
            - ``"growth_mb"``: peak − initial.
            - ``"growth_rate_mb_per_min"``: Linear growth rate.
            - ``"is_leaking"``: True if growth > 10% of initial.
    """
    import numpy as np

    if not memory_samples_mb:
        return {
            "initial_mb": 0.0,
            "peak_mb": 0.0,
            "final_mb": 0.0,
            "growth_mb": 0.0,
            "growth_rate_mb_per_min": 0.0,
            "is_leaking": False,
        }
    arr = np.array(memory_samples_mb, dtype=float)
    initial = float(arr[0])
    peak = float(arr.max())
    final = float(arr[-1])
    growth = peak - initial

    rate = 0.0
    if len(arr) >= 2:
        x_minutes = np.arange(len(arr)) * interval_seconds / 60.0
        slope = float(np.polyfit(x_minutes, arr, 1)[0])
        rate = slope

    return {
        "initial_mb": initial,
        "peak_mb": peak,
        "final_mb": final,
        "growth_mb": growth,
        "growth_rate_mb_per_min": rate,
        "is_leaking": growth > 0.1 * max(initial, 1.0),
    }


def session_length_statistical_comparison(
    metrics_by_duration: dict[str, list[float]],
) -> dict[str, Any]:
    """Compare a metric across session durations using ANOVA + pairwise Cohen's d.

    Args:
        metrics_by_duration: Mapping from duration label (e.g. "5min")
            to list of metric values (one per seed/trial).

    Returns:
        Dict with:
            - ``"anova_f"``: F-statistic (or Kruskal-Wallis H if < 3 groups).
            - ``"anova_p"``: p-value.
            - ``"pairwise_d"``: Dict of pairwise Cohen's d values.
            - ``"significant"``: True if p < 0.05.
            - ``"eta_squared"``: Effect size (proportion of variance).
    """
    import numpy as np

    groups = list(metrics_by_duration.values())
    labels = list(metrics_by_duration.keys())

    if len(groups) < 2:
        return {
            "anova_f": 0.0,
            "anova_p": 1.0,
            "pairwise_d": {},
            "significant": False,
            "eta_squared": 0.0,
        }

    # One-way ANOVA (manual — avoid scipy dependency)
    all_vals = []
    for g in groups:
        all_vals.extend(g)
    grand_mean = float(np.mean(all_vals))
    n_total = len(all_vals)
    k = len(groups)

    ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups)
    ss_within = sum(sum((v - float(np.mean(g))) ** 2 for v in g) for g in groups)
    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = n_total - k

    if df_within <= 0 or ss_within == 0:
        return {
            "anova_f": float("inf") if ss_between > 0 else 0.0,
            "anova_p": 0.0 if ss_between > 0 else 1.0,
            "pairwise_d": {},
            "significant": ss_between > 0,
            "eta_squared": 1.0 if ss_total > 0 and ss_between > 0 else 0.0,
        }

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within

    # Approximate p-value using F-distribution CDF (simplified)
    # For rigorous p, would need scipy.stats.f — use a conservative
    # threshold approximation.
    # Critical F at p=0.05 for df1=k-1, df2=n-k:
    # Use the approximation that F > 4.0 with df >= 3 per group
    # is typically p < 0.05
    # For more exact results we compute via the regularised
    # incomplete beta function (manual impl to avoid scipy).
    p_val = _f_distribution_p_value(f_stat, df_between, df_within)

    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    # Pairwise Cohen's d
    pairwise_d: dict[str, float] = {}
    for i in range(k):
        for j in range(i + 1, k):
            g_a = np.array(groups[i], dtype=float)
            g_b = np.array(groups[j], dtype=float)
            pooled_std = float(
                np.sqrt(
                    (
                        (len(g_a) - 1) * g_a.var(ddof=1)
                        + (len(g_b) - 1) * g_b.var(ddof=1)
                    )
                    / max(len(g_a) + len(g_b) - 2, 1)
                )
            )
            d = (float(g_a.mean()) - float(g_b.mean())) / max(pooled_std, 1e-9)
            pairwise_d[f"{labels[i]}_vs_{labels[j]}"] = round(d, 4)

    return {
        "anova_f": round(f_stat, 4),
        "anova_p": round(p_val, 6),
        "pairwise_d": pairwise_d,
        "significant": p_val < 0.05,
        "eta_squared": round(eta_sq, 4),
    }


def _f_distribution_p_value(
    f_stat: float,
    df1: int,
    df2: int,
) -> float:
    """Approximate p-value for F-distribution using the regularised
    incomplete beta function.

    Computes P(F > f_stat) = 1 - I_x(a, b) where:
        x = df1 * f / (df1 * f + df2)
        a = df1 / 2
        b = df2 / 2
    """
    if f_stat <= 0:
        return 1.0
    x = df1 * f_stat / (df1 * f_stat + df2)
    a = df1 / 2.0
    b = df2 / 2.0
    return 1.0 - _regularised_incomplete_beta(x, a, b)


def _regularised_incomplete_beta(
    x: float,
    a: float,
    b: float,
    max_iter: int = 200,
) -> float:
    """Regularised incomplete beta function I_x(a, b) via continued fraction.

    Uses Lentz's method for the continued fraction representation.
    """
    import math as _math

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use symmetry relation for numerical stability
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_incomplete_beta(1.0 - x, b, a, max_iter)

    # Log of the prefactor
    ln_prefix = (
        _math.lgamma(a + b)
        - _math.lgamma(a)
        - _math.lgamma(b)
        + a * _math.log(x)
        + b * _math.log(1.0 - x)
    )
    prefix = _math.exp(ln_prefix)

    # Continued fraction (Lentz's method)
    tiny = 1e-30
    f = tiny
    c = tiny
    d = 0.0

    for m in range(max_iter):
        # a_m coefficient
        if m == 0:
            a_m = 1.0
        elif m % 2 == 1:
            k = (m - 1) // 2 + 1
            a_m = (
                -(a + k - 1.0 + k)
                * (a + k - 1.0)
                * x
                / ((a + 2 * k - 2.0) * (a + 2 * k - 1.0))
            )
        else:
            k = m // 2
            a_m = k * (b - k) * x / ((a + 2 * k - 1.0) * (a + 2 * k))

        d = 1.0 + a_m * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d

        c = 1.0 + a_m / c
        if abs(c) < tiny:
            c = tiny

        f *= c * d

        if abs(c * d - 1.0) < 1e-12:
            break

    return prefix * (f - 1.0) / a


# =========================================================================
# Y4 Q1.8: Noise Tolerance, Parameter-Matched, Curriculum, Community
# =========================================================================


class PhaseTrackerLarge(nn.Module):
    """Scaled-up PhaseTracker targeting ~84K params (matching SlotAttention).

    Increases oscillator count (28->112), widens MLPs (hidden 64->192),
    and adds a 2-layer residual dynamics MLP for richer phase evolution.

    Args:
        detection_dim: Per-detection feature dimension.
        n_osc: Total oscillators (default 112 for 16+32+64 bands).
        hidden_dim: MLP hidden dimension.
        n_discrete_steps: Dynamics integration steps.
        match_threshold: Minimum phase similarity for valid match.
    """

    def __init__(
        self,
        detection_dim: int = 4,
        n_osc: int = 112,
        hidden_dim: int = 192,
        n_discrete_steps: int = 5,
        match_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        self.n_osc = n_osc
        self._n_discrete_steps = n_discrete_steps
        self.match_threshold = match_threshold

        # Wider MLPs
        self.det_to_phase = nn.Sequential(
            nn.Linear(detection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_osc),
        )
        self.det_to_amp = nn.Sequential(
            nn.Linear(detection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_osc),
            nn.Softplus(),
        )

        # Band split: 16 delta + 32 theta + 64 gamma = 112
        n_delta = n_osc // 7  # 16
        n_theta = n_osc * 2 // 7  # 32
        n_gamma = n_osc - n_delta - n_theta  # 64

        self.dynamics = DiscreteDeltaThetaGamma(
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
        )

        # Residual dynamics MLP for richer phase evolution
        self.phase_refine = nn.Sequential(
            nn.Linear(n_osc, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_osc),
        )

    def encode(self, detections: Tensor) -> tuple[Tensor, Tensor]:
        """Encode detections to (phase, amplitude)."""
        phase_raw = self.det_to_phase(detections)
        phase = phase_raw % (2.0 * math.pi)
        amp = self.det_to_amp(detections)
        return phase, amp

    def evolve(self, phase: Tensor, amplitude: Tensor) -> tuple[Tensor, Tensor]:
        """Evolve phase state through dynamics + residual refinement."""
        phase, amplitude = self.dynamics.integrate(
            phase,
            amplitude,
            n_steps=self._n_discrete_steps,
            dt=0.01,
        )
        # Residual refinement
        phase = phase + 0.1 * self.phase_refine(phase)
        phase = phase % (2.0 * math.pi)
        return phase, amplitude

    def phase_similarity(self, phase_a: Tensor, phase_b: Tensor) -> Tensor:
        """Phase similarity matrix via cosine of complex embeddings."""
        _EPS = 1e-8
        z_a = torch.exp(1j * phase_a.to(torch.complex64))
        z_b = torch.exp(1j * phase_b.to(torch.complex64))
        z_a_norm = z_a / (z_a.abs().pow(2).sum(dim=-1, keepdim=True).sqrt() + _EPS)
        z_b_norm = z_b / (z_b.abs().pow(2).sum(dim=-1, keepdim=True).sqrt() + _EPS)
        sim = (
            (z_a_norm.unsqueeze(1) * z_b_norm.conj().unsqueeze(0))
            .sum(dim=-1)
            .real.float()
        )
        return sim

    def forward(
        self,
        detections_t: Tensor,
        detections_t1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Match detections across two consecutive frames."""
        phase_t, amp_t = self.encode(detections_t)
        phase_t1, amp_t1 = self.encode(detections_t1)
        phase_t_evolved, _ = self.evolve(phase_t, amp_t)
        sim = self.phase_similarity(phase_t_evolved, phase_t1)
        N_t = detections_t.shape[0]
        matches = torch.full((N_t,), -1, dtype=torch.long, device=detections_t.device)
        used = torch.zeros(
            detections_t1.shape[0], dtype=torch.bool, device=detections_t.device
        )
        max_sims, max_idxs = sim.max(dim=1)
        order = max_sims.argsort(descending=True)
        for idx in order:
            best_j = int(max_idxs[idx].item())
            if not used[best_j] and max_sims[idx].item() >= self.match_threshold:
                matches[idx] = best_j
                used[best_j] = True
        return matches, sim

    def track_sequence(self, frame_detections: list[Tensor]) -> dict[str, Any]:
        """Track objects across a sequence of frames."""
        if len(frame_detections) < 2:
            return {
                "phase_history": [],
                "identity_matches": [],
                "identity_preservation": 1.0,
                "per_frame_similarity": [],
                "per_frame_phase_correlation": [],
            }
        all_matches = []
        all_sims = []
        all_corrs = []
        for t in range(len(frame_detections) - 1):
            matches, sim = self(frame_detections[t], frame_detections[t + 1])
            all_matches.append(matches)
            all_sims.append(float(sim.max(dim=1).values.mean().item()))
            diag_sim = torch.diagonal(
                sim[
                    : min(sim.shape[0], sim.shape[1]), : min(sim.shape[0], sim.shape[1])
                ]
            )
            all_corrs.append(
                float(diag_sim.mean().item()) if diag_sim.numel() > 0 else 0.0
            )
        # Identity preservation
        n_correct = 0
        n_total = 0
        for m in all_matches:
            for i, j in enumerate(m):
                n_total += 1
                if j.item() == i:
                    n_correct += 1
        ip = n_correct / max(n_total, 1)
        return {
            "phase_history": [],
            "identity_matches": all_matches,
            "identity_preservation": ip,
            "per_frame_similarity": all_sims,
            "per_frame_phase_correlation": all_corrs,
        }


def noise_tolerance_sweep(
    pt_model: nn.Module,
    sa_model: nn.Module,
    dataset_fn: Any,
    sigmas: list[float],
    n_seeds: int = 3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Sweep noise levels and compare IP for PT vs SA.

    Args:
        pt_model: PhaseTracker model.
        sa_model: SlotAttention model.
        dataset_fn: Callable(seed, noise_sigma) -> dataset list.
        sigmas: Noise standard deviations to test.
        n_seeds: Number of random seeds per sigma.
        device: Device.

    Returns:
        Dict with per-sigma results for both models.
    """
    results: dict[str, Any] = {}
    for sigma in sigmas:
        pt_ips: list[float] = []
        sa_ips: list[float] = []
        for s in range(n_seeds):
            ds = dataset_fn(seed=42 + s, noise_sigma=sigma)
            for model, ips in [(pt_model, pt_ips), (sa_model, sa_ips)]:
                model.eval()
                model.to(device)
                seq_ips: list[float] = []
                m: Any = model
                with torch.no_grad():
                    for seq in ds:
                        frames = [f.to(device) for f in seq.frames]
                        res = m.track_sequence(frames)
                        seq_ips.append(res["identity_preservation"])
                ips.append(sum(seq_ips) / max(len(seq_ips), 1))
        results[str(sigma)] = {
            "sigma": sigma,
            "pt_ips": pt_ips,
            "sa_ips": sa_ips,
            "pt_mean": sum(pt_ips) / max(len(pt_ips), 1),
            "sa_mean": sum(sa_ips) / max(len(sa_ips), 1),
            "pt_ci": bootstrap_ci(pt_ips) if len(pt_ips) >= 2 else None,
            "sa_ci": bootstrap_ci(sa_ips) if len(sa_ips) >= 2 else None,
        }
    return results


def noise_degradation_curve(
    model: nn.Module,
    dataset_fn: Any,
    sigmas: list[float],
    n_seeds: int = 3,
    device: str = "cpu",
) -> dict[str, Any]:
    """IP as function of noise sigma with bootstrap CI.

    Args:
        model: Tracker model.
        dataset_fn: Callable(seed, noise_sigma) -> dataset list.
        sigmas: Noise levels.
        n_seeds: Seeds.
        device: Device.

    Returns:
        Dict with per-sigma IP and CI.
    """
    import numpy as np

    curve: dict[str, Any] = {}
    for sigma in sigmas:
        ips: list[float] = []
        for s in range(n_seeds):
            ds = dataset_fn(seed=42 + s, noise_sigma=sigma)
            model.eval()
            model.to(device)
            seq_ips: list[float] = []
            m: Any = model
            with torch.no_grad():
                for seq in ds:
                    frames = [f.to(device) for f in seq.frames]
                    res = m.track_sequence(frames)
                    seq_ips.append(res["identity_preservation"])
            ips.append(sum(seq_ips) / max(len(seq_ips), 1))
        curve[str(sigma)] = {
            "sigma": sigma,
            "ips": ips,
            "mean": float(np.mean(ips)),
            "std": float(np.std(ips, ddof=1)) if len(ips) > 1 else 0.0,
            "ci": bootstrap_ci(ips) if len(ips) >= 2 else None,
        }
    return curve


def noise_crossover_analysis(
    pt_curve: dict[str, Any],
    sa_curve: dict[str, Any],
) -> dict[str, Any]:
    """Find crossover sigma where PT IP > SA IP.

    Args:
        pt_curve: From noise_degradation_curve for PT.
        sa_curve: From noise_degradation_curve for SA.

    Returns:
        Dict with crossover sigma, degradation rates, stats.
    """
    import numpy as np

    sigmas = sorted([float(k) for k in pt_curve.keys()])
    pt_means = [pt_curve[str(s)]["mean"] for s in sigmas]
    sa_means = [sa_curve[str(s)]["mean"] for s in sigmas]

    # Find crossover
    crossover_sigma = None
    for i in range(len(sigmas) - 1):
        diff_i = pt_means[i] - sa_means[i]
        diff_j = pt_means[i + 1] - sa_means[i + 1]
        if diff_i <= 0 and diff_j > 0:
            # Linear interpolation
            f = -diff_i / max(diff_j - diff_i, 1e-12)
            crossover_sigma = sigmas[i] + f * (sigmas[i + 1] - sigmas[i])
            break

    # Fit exponential decay: IP = IP0 * exp(-lambda * sigma)
    def _fit_exp(means: list[float]) -> float:
        s_arr = np.array(sigmas)
        m_arr = np.clip(np.array(means), 1e-10, None)
        ln_m = np.log(m_arr)
        if len(s_arr) >= 2:
            coeffs = np.polyfit(s_arr, ln_m, 1)
            return -float(coeffs[0])
        return 0.0

    lambda_pt = _fit_exp(pt_means)
    lambda_sa = _fit_exp(sa_means)

    # Statistical tests at each sigma
    stats = {}
    for s in sigmas:
        sk = str(s)
        if sk in pt_curve and sk in sa_curve:
            pt_ips = pt_curve[sk].get("ips", [])
            sa_ips = sa_curve[sk].get("ips", [])
            if len(pt_ips) >= 2 and len(sa_ips) >= 2:
                stats[sk] = welch_t_test(pt_ips, sa_ips)

    return {
        "crossover_sigma": crossover_sigma,
        "lambda_pt": lambda_pt,
        "lambda_sa": lambda_sa,
        "pt_degrades_slower": lambda_pt < lambda_sa,
        "per_sigma_stats": stats,
    }


def curriculum_dataset(
    stage: int,
    n_seqs: int = 20,
    det_dim: int = 4,
    seed: int = 42,
) -> list[Any]:
    """Generate dataset for a curriculum stage.

    Stage 1: 2 objects, T=10
    Stage 2: 3 objects, T=20
    Stage 3: 4 objects, T=40
    Stage 4: 6 objects, T=60

    Args:
        stage: Stage number (1-4).
        n_seqs: Number of sequences.
        det_dim: Detection feature dimension.
        seed: Random seed.

    Returns:
        List of SequenceData.
    """
    from prinet.utils.temporal_training import generate_dataset

    stage_config = {
        1: (2, 10),
        2: (3, 20),
        3: (4, 40),
        4: (6, 60),
    }
    n_obj, n_frames = stage_config.get(stage, (4, 20))
    return generate_dataset(
        n_seqs,
        n_objects=n_obj,
        n_frames=n_frames,
        det_dim=det_dim,
        base_seed=seed,
    )


def curriculum_train(
    model: nn.Module,
    n_stages: int = 4,
    epochs_per_stage: int = 10,
    n_train: int = 30,
    n_val: int = 10,
    det_dim: int = 4,
    lr: float = 3e-4,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Train model through progressive difficulty stages.

    Args:
        model: Tracker model.
        n_stages: Number of curriculum stages (1-4).
        epochs_per_stage: Epochs per stage.
        n_train: Training sequences per stage.
        n_val: Validation sequences per stage.
        det_dim: Detection dimension.
        lr: Learning rate.
        device: Device.
        seed: Random seed.

    Returns:
        Dict with per-stage convergence info.
    """
    from prinet.utils.temporal_training import TemporalTrainer

    results = {}
    for stage in range(1, n_stages + 1):
        train_ds = curriculum_dataset(stage, n_train, det_dim, seed + stage * 100)
        val_ds = curriculum_dataset(stage, n_val, det_dim, seed + stage * 200)

        trainer = TemporalTrainer(
            model=model,
            lr=lr,
            max_epochs=epochs_per_stage,
            patience=epochs_per_stage,
            device=device,
        )
        tr = trainer.train(train_data=train_ds, val_data=val_ds)

        results[f"stage_{stage}"] = {
            "n_objects": [2, 3, 4, 6][stage - 1],
            "n_frames": [10, 20, 40, 60][stage - 1],
            "final_val_ip": tr.final_val_ip,
            "best_epoch": tr.best_epoch,
            "total_epochs": tr.total_epochs,
            "wall_time_s": tr.wall_time_s,
            "val_ips": tr.val_ips,
        }

    return results


def per_community_order_parameter(
    phase: Tensor,
    community_assignments: list[list[int]],
) -> list[float]:
    """Compute Kuramoto r per community.

    Args:
        phase: Phase tensor ``(N,)``.
        community_assignments: List of index lists, one per community.

    Returns:
        List of per-community r values.
    """
    r_values = []
    for indices in community_assignments:
        if not indices:
            r_values.append(0.0)
            continue
        idx_t = torch.tensor(indices, dtype=torch.long, device=phase.device)
        sub_phase = phase[idx_t]
        z = torch.exp(1j * sub_phase.to(torch.complex64))
        r = float(z.mean().abs().item())
        r_values.append(r)
    return r_values
