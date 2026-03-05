"""PyTorch Neural Network Layers for PRINet.

Provides ``nn.Module``-compatible layers that wrap oscillator dynamics
for use in standard PyTorch training pipelines. Implements Task 1.6
(ResonanceLayer integration), oscillatory weight initialization, and
numerical stability safeguards (Q2 P0 NaN fix).

Example:
    >>> import torch
    >>> layer = ResonanceLayer(n_oscillators=64, n_dims=128)
    >>> x = torch.randn(32, 128)  # Batch of 32
    >>> output = layer(x)
    >>> print(output.shape)
    torch.Size([32, 64])
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prinet.core.measurement import (
    extract_concept_probabilities,
    kuramoto_order_parameter,
    power_spectral_density,
)

# ---------------------------------------------------------------------------
# Constants for numerical stability (P0 NaN fix)
# ---------------------------------------------------------------------------

_PHASE_WRAP: float = 2.0 * math.pi
"""Wrap phases to [0, 2π) to prevent unbounded growth."""

_AMP_MAX: float = 10.0
"""Upper clamp for oscillator amplitudes to prevent explosion."""

_EPS: float = 1e-6
"""Epsilon guard for divisions and log-softmax."""


def _wrap_phase(phase: Tensor) -> Tensor:
    """Wrap phase tensor to [0, 2π).

    Args:
        phase: Phase tensor of any shape.

    Returns:
        Wrapped phase in [0, 2π).
    """
    return phase % _PHASE_WRAP


class ResonanceLayer(nn.Module):
    """Single resonance propagation layer as a PyTorch Module.

    Wraps extended Kuramoto dynamics into a differentiable layer that
    can be composed with standard ``nn.Module`` components. Converts
    input features to oscillator states, propagates dynamics, and
    returns amplitude-phase representations.

    Args:
        n_oscillators: Number of coupled oscillators ``R``.
        n_dims: Input feature dimension ``D``.
        n_steps: Number of ODE integration steps per forward pass.
        dt: Timestep for ODE integration.
        decay_rate: Base amplitude decay rate.
        freq_adaptation_rate: Frequency adaptation rate γ.

    Example:
        >>> layer = ResonanceLayer(n_oscillators=64, n_dims=256)
        >>> x = torch.randn(16, 256)
        >>> out = layer(x)
        >>> print(out.shape)
        torch.Size([16, 64])
    """

    def __init__(
        self,
        n_oscillators: int,
        n_dims: int = 256,
        n_steps: int = 10,
        dt: float = 0.01,
        decay_rate: float = 0.1,
        freq_adaptation_rate: float = 0.01,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.n_oscillators = n_oscillators
        self.n_dims = n_dims
        self.n_steps = n_steps
        self.dt = dt
        self._decay_rate = decay_rate
        self._gamma = freq_adaptation_rate
        self._compile = compile

        # Coupling scale factor 1/√N for numerical stability (P0 NaN fix)
        self._coupling_scale: float = 1.0 / math.sqrt(n_oscillators)

        # Learnable coupling matrix (N, N)
        self.coupling = nn.Parameter(
            torch.randn(n_oscillators, n_oscillators) * 0.1
        )

        # Learnable decay rates per oscillator
        self.decay = nn.Parameter(
            torch.full((n_oscillators,), decay_rate)
        )

        # Input projection: maps input features to initial oscillator states
        self.input_proj = nn.Linear(n_dims, n_oscillators, bias=False)

        # Frequency modulation parameters
        self.modulation = nn.Parameter(
            torch.randn(n_oscillators, n_oscillators) * 0.01
        )

        # Base frequencies initialized as linearly spaced
        self.base_frequency = nn.Parameter(
            torch.linspace(0.1, 10.0, n_oscillators)
        )

        # Initialize with oscillatory-aware scheme
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using oscillatory-network-aware scheme.

        Coupling is initialized with small symmetric values to promote
        stable synchronization. Input projection uses Xavier scaling
        adapted for phase-space mapping.
        """
        # Symmetric coupling initialization for stability
        with torch.no_grad():
            sym = (self.coupling + self.coupling.T) / 2.0
            self.coupling.copy_(sym * 0.1)
            self.coupling.fill_diagonal_(0.0)

        # Xavier-like initialization scaled for oscillators
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)

    def _compute_initial_state(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Project input features to initial oscillator state.

        Args:
            x: Input tensor of shape ``(B, D)`` or ``(D,)``.

        Returns:
            Tuple of ``(phase, amplitude, frequency)`` each of shape
            ``(B, R)`` or ``(R,)``.
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        # Project input to oscillator space
        projection = self.input_proj(x)  # (B, R)

        # Initialize phase from FFT of projection
        z_complex = torch.fft.rfft(projection, dim=-1)
        n_freq = z_complex.shape[-1]

        # Pad or truncate to n_oscillators
        phase = torch.zeros_like(projection)
        amp = torch.zeros_like(projection)

        phase_vals = torch.angle(z_complex)
        amp_vals = torch.abs(z_complex) + _EPS

        # Map FFT components to oscillators
        n_use = min(n_freq, self.n_oscillators)
        phase[:, :n_use] = phase_vals[:, :n_use]
        amp[:, :n_use] = amp_vals[:, :n_use]

        # Fill remaining oscillators with input-derived values
        if n_use < self.n_oscillators:
            amp[:, n_use:] = torch.abs(projection[:, n_use:]) + _EPS
            phase[:, n_use:] = torch.atan2(
                projection[:, n_use:],
                torch.abs(projection[:, n_use:]) + _EPS,
            )

        # Clamp initial amplitudes & wrap phases (P0 NaN fix)
        amp = torch.clamp(amp, min=_EPS, max=_AMP_MAX)
        phase = _wrap_phase(phase)

        frequency = self.base_frequency.unsqueeze(0).expand_as(projection)

        if was_1d:
            return phase.squeeze(0), amp.squeeze(0), frequency.squeeze(0)
        return phase, amp, frequency

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the resonance layer.

        Converts input to oscillator state, runs Kuramoto dynamics
        for ``n_steps``, and returns the final amplitude representation.

        Args:
            x: Input tensor of shape ``(B, D)`` or ``(D,)``.

        Returns:
            Output tensor of shape ``(B, R)`` or ``(R,)`` representing
            the oscillator amplitudes after propagation.
        """
        phase, amplitude, frequency = self._compute_initial_state(x)

        # Ensure coupling diagonal is zero (no self-coupling)
        coupling = self.coupling.clone()
        coupling.fill_diagonal_(0.0)
        # Scale coupling by 1/√N for numerical stability (P0 NaN fix)
        coupling = coupling * self._coupling_scale

        for _ in range(self.n_steps):
            # Phase differences: φⱼ - φᵢ
            phase_diff = (
                phase.unsqueeze(-2) - phase.unsqueeze(-1)
            )  # (..., R, R)

            sin_diff = torch.sin(phase_diff)
            cos_diff = torch.cos(phase_diff)

            # Amplitude-weighted coupling
            amp_weight = amplitude.unsqueeze(-2)

            # Phase update: dφᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(φⱼ - φᵢ) |rⱼ|
            phase_update = (coupling * sin_diff * amp_weight).sum(dim=-1)
            phase = phase + self.dt * (frequency + phase_update)
            # Wrap phase to [0, 2π) after every step (P0 NaN fix)
            phase = _wrap_phase(phase)

            # Amplitude update: drᵢ/dt = -λᵢ rᵢ + Σⱼ Kᵢⱼ cos(...) |rⱼ|
            amp_coupling = (coupling * cos_diff * amp_weight).sum(dim=-1)
            amplitude = torch.clamp(
                amplitude
                + self.dt * (-self.decay * amplitude + amp_coupling),
                min=_EPS,
                max=_AMP_MAX,
            )

            # Frequency modulation (slow)
            freq_update = (self.modulation * sin_diff * amp_weight).sum(
                dim=-1
            )
            frequency = frequency + self.dt * self._gamma * freq_update

        return amplitude

    def get_order_parameter(self, x: Tensor) -> Tensor:
        """Compute the Kuramoto order parameter for the given input.

        Useful for monitoring synchronization during training.

        Args:
            x: Input tensor of shape ``(B, D)`` or ``(D,)``.

        Returns:
            Order parameter(s) in ``[0, 1]``.
        """
        phase, _, _ = self._compute_initial_state(x)

        coupling = self.coupling.clone()
        coupling.fill_diagonal_(0.0)
        coupling = coupling * self._coupling_scale

        amplitude = torch.ones_like(phase)
        frequency = self.base_frequency.unsqueeze(0).expand_as(phase)

        for _ in range(self.n_steps):
            phase_diff = phase.unsqueeze(-2) - phase.unsqueeze(-1)
            sin_diff = torch.sin(phase_diff)
            amp_weight = amplitude.unsqueeze(-2)
            phase_update = (coupling * sin_diff * amp_weight).sum(dim=-1)
            phase = _wrap_phase(
                phase + self.dt * (frequency + phase_update)
            )

        return kuramoto_order_parameter(phase)


class PRINetModel(nn.Module):
    """Full PRINet model with decomposition, propagation, and measurement.

    Combines polyadic input decomposition, multiple resonance propagation
    layers, and spectral measurement into an end-to-end differentiable
    model for concept classification.

    Args:
        n_resonances: Number of oscillators per layer.
        n_dims: Input feature dimension.
        n_concepts: Number of output concepts/classes.
        n_layers: Number of resonance propagation layers.
        n_steps: ODE integration steps per layer.
        dt: ODE timestep.
        compile: If ``True``, wrap the model with
            ``torch.compile(mode="reduce-overhead")`` for 2-6× GPU
            speedup via CUDA Graph fusion.

    Example:
        >>> model = PRINetModel(
        ...     n_resonances=64, n_dims=256, n_concepts=10
        ... )
        >>> x = torch.randn(32, 256)
        >>> probs = model(x)
        >>> print(probs.shape)
        torch.Size([32, 10])
    """

    def __init__(
        self,
        n_resonances: int = 64,
        n_dims: int = 256,
        n_concepts: int = 10,
        n_layers: int = 4,
        n_steps: int = 10,
        dt: float = 0.01,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.n_resonances = n_resonances
        self.n_dims = n_dims
        self.n_concepts = n_concepts
        self._use_compile = compile
        self._mixed_precision: bool = False
        self._amp_dtype: torch.dtype = torch.bfloat16

        # Input projection to oscillator space
        self.input_layer = ResonanceLayer(
            n_oscillators=n_resonances,
            n_dims=n_dims,
            n_steps=n_steps,
            dt=dt,
        )

        # Stacked resonance layers
        self.layers = nn.ModuleList(
            [
                ResonanceLayer(
                    n_oscillators=n_resonances,
                    n_dims=n_resonances,
                    n_steps=n_steps,
                    dt=dt,
                )
                for _ in range(n_layers - 1)
            ]
        )

        # Layer normalization for stability between resonance layers
        # (P0 NaN fix — normalizes amplitudes before next layer)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(n_resonances) for _ in range(n_layers)]
        )

        # Concept readout: learned frequency-to-class projection
        self.concept_proj = nn.Linear(n_resonances, n_concepts)

        # Apply torch.compile for GPU kernel fusion (Q2 optimization)
        if compile and hasattr(torch, "compile"):
            self._apply_compile()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through full PRINet.

        Includes numerical stability safeguards (P0 NaN fix):
        - LayerNorm between resonance layers to prevent amplitude drift
        - Logit clamping before log_softmax to prevent overflow
        - Phase wrapping and amplitude clamping in each ResonanceLayer
        - Optional mixed-precision via ``torch.autocast`` (Q2)

        Args:
            x: Input tensor of shape ``(B, D)`` or ``(D,)``.

        Returns:
            Log-probabilities of shape ``(B, K)`` or ``(K,)``
            over K concepts.
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        # Determine device type for autocast
        device_type = x.device.type if x.device.type != "cpu" else "cpu"

        # Use autocast context if mixed precision is enabled
        ctx = (
            torch.autocast(device_type=device_type, dtype=self._amp_dtype)
            if self._mixed_precision
            else torch.autocast(device_type=device_type, enabled=False)
        )

        with ctx:
            # Propagate through resonance layers with LayerNorm
            h = self.input_layer(x)
            h = self.layer_norms[0](h)

            for i, layer in enumerate(self.layers):
                h = layer(h)
                h = self.layer_norms[i + 1](h)

        # Always compute logits + log_softmax in float32 for stability
        h = h.float()
        logits = self.concept_proj(h)
        # Clamp logits to prevent log_softmax overflow (P0 NaN fix)
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        log_probs = torch.log_softmax(logits, dim=-1)

        if was_1d:
            return log_probs.squeeze(0)
        return log_probs

    def _apply_compile(self) -> None:
        """Apply torch.compile to sub-modules for GPU kernel fusion.

        Uses ``reduce-overhead`` mode which enables CUDA Graphs,
        bundling all RK4 integration steps into a single GPU launch.
        """
        # Compile individual resonance layers for maximum fusion
        self.input_layer = torch.compile(  # type: ignore[assignment]
            self.input_layer, mode="reduce-overhead"
        )
        for i, layer in enumerate(self.layers):
            self.layers[i] = torch.compile(  # type: ignore[assignment]
                layer, mode="reduce-overhead"
            )

    def enable_mixed_precision(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "PRINetModel":
        """Enable or disable automatic mixed precision.

        When enabled, the forward pass runs under ``torch.autocast``
        with the specified dtype. ``bfloat16`` is recommended over
        ``float16`` for its larger dynamic range, which prevents
        overflow in oscillator dynamics.

        Args:
            enabled: Whether to enable mixed precision.
            dtype: The reduced-precision dtype. Default ``bfloat16``.

        Returns:
            Self for method chaining.
        """
        self._mixed_precision = enabled
        self._amp_dtype = dtype
        return self


def compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",
) -> nn.Module:
    """Wrap a PRINet model or layer with ``torch.compile``.

    This is the recommended way to enable GPU kernel fusion for
    PRINet models. The ``reduce-overhead`` mode uses CUDA Graphs
    to bundle many small kernel launches (e.g. 200 RK4 steps)
    into a single GPU operation, yielding 2-6× speedup.

    Args:
        model: Any ``nn.Module`` (typically ``PRINetModel`` or
            ``ResonanceLayer``).
        mode: Compilation mode. ``"reduce-overhead"`` is recommended
            for PRINet's iterative ODE loops.

    Returns:
        Compiled module.

    Example:
        >>> model = PRINetModel(n_resonances=64, n_dims=256)
        >>> compiled = compile_model(model)
    """
    if not hasattr(torch, "compile"):
        return model
    return torch.compile(model, mode=mode)  # type: ignore[return-value]


def oscillatory_weight_init(
    module: nn.Module,
    coupling_scale: float = 0.1,
    proj_gain: float = 0.5,
) -> None:
    """Apply oscillatory-network-aware weight initialization.

    Initializes coupling matrices symmetrically with small values,
    linear projections with adapted Xavier scaling, and biases to
    zero. This promotes stable initial synchronization.

    Args:
        module: The module to initialize.
        coupling_scale: Scale for coupling matrix initialization.
        proj_gain: Gain for Xavier uniform initialization.

    Example:
        >>> model = PRINetModel()
        >>> oscillatory_weight_init(model)
    """
    for name, param in module.named_parameters():
        if "coupling" in name and param.dim() == 2:
            with torch.no_grad():
                sym = (param + param.T) / 2.0
                param.copy_(sym * coupling_scale)
                if param.shape[0] == param.shape[1]:
                    param.fill_diagonal_(0.0)
        elif "weight" in name and param.dim() == 2:
            nn.init.xavier_uniform_(param, gain=proj_gain)
        elif "bias" in name:
            nn.init.zeros_(param)


# =========================================================================
# Q3: Hierarchical & Phase-to-Rate Neural Network Layers
# =========================================================================


class PhaseToRateConverter(nn.Module):
    """Neural network module for phase-amplitude to rate-code conversion.

    Wraps :func:`prinet.core.propagation.phase_to_rate` as a
    differentiable ``nn.Module`` with a learnable temperature parameter.

    Supports three WTA modes:
        - ``"soft"``: Fully differentiable softmax-based.
        - ``"hard"``: Top-k selection (sparse but non-differentiable).
        - ``"annealed"``: Temperature-driven interpolation.

    Args:
        n_oscillators: Input oscillator dimension.
        mode: WTA mode (``"soft"``, ``"hard"``, ``"annealed"``).
        sparsity: Target fraction of active units.
        initial_temperature: Starting temperature for soft/annealed.
        learnable_temperature: If ``True``, temperature is an
            ``nn.Parameter``.

    Example:
        >>> converter = PhaseToRateConverter(64, mode="soft")
        >>> phase = torch.rand(32, 64) * 2 * 3.14159
        >>> amp = torch.ones(32, 64)
        >>> rates = converter(phase, amp)
        >>> print(rates.shape)
        torch.Size([32, 64])
    """

    def __init__(
        self,
        n_oscillators: int,
        mode: str = "soft",
        sparsity: float = 0.1,
        initial_temperature: float = 1.0,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        self.n_oscillators = n_oscillators
        self._mode = mode
        self._sparsity = sparsity

        if learnable_temperature:
            self.temperature = nn.Parameter(
                torch.tensor(initial_temperature)
            )
        else:
            self.register_buffer(
                "temperature", torch.tensor(initial_temperature)
            )

    @property
    def mode(self) -> str:
        """Current WTA mode."""
        return self._mode

    @property
    def sparsity(self) -> float:
        """Target sparsity fraction."""
        return self._sparsity

    def forward(self, phase: Tensor, amplitude: Tensor) -> Tensor:
        """Convert phase-amplitude to rate codes.

        Args:
            phase: Phase tensor ``(B, N)`` or ``(N,)``.
            amplitude: Amplitude tensor, same shape.

        Returns:
            Rate-coded output, same shape. Non-negative.
        """
        from prinet.core.propagation import phase_to_rate

        temp = torch.clamp(self.temperature, min=1e-6).item()
        return phase_to_rate(
            phase,
            amplitude,
            mode=self._mode,
            sparsity=self._sparsity,
            temperature=temp,
        )


class SparsityRegularizationLoss(nn.Module):
    """Sigmoid-surrogate L0 sparsity regularization loss.

    Encourages a target sparsity level in rate-coded activations.
    Uses a differentiable sigmoid surrogate for the L0 norm:

        L = |actual_sparsity - target_sparsity|²

    where ``actual_sparsity = 1 - mean(σ(x / τ))`` and ``τ`` is a
    temperature controlling the sharpness of the threshold.

    Args:
        target_sparsity: Target fraction of zero (inactive) units.
        temperature: Sigmoid temperature for surrogate.

    Example:
        >>> loss_fn = SparsityRegularizationLoss(target_sparsity=0.9)
        >>> rates = torch.rand(32, 64)
        >>> loss = loss_fn(rates)
    """

    def __init__(
        self,
        target_sparsity: float = 0.9,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self._target = target_sparsity
        self._temperature = temperature

    @property
    def target_sparsity(self) -> float:
        """Target sparsity level."""
        return self._target

    def forward(self, activations: Tensor) -> Tensor:
        """Compute sparsity regularization loss.

        Args:
            activations: Rate-coded activations ``(..., N)``.

        Returns:
            Scalar loss tensor.
        """
        # Sigmoid surrogate for "is this unit active?"
        active_prob = torch.sigmoid(activations / self._temperature)
        actual_density = active_prob.mean()
        actual_sparsity = 1.0 - actual_density
        return (actual_sparsity - self._target) ** 2


class HierarchicalResonanceLayer(nn.Module):
    """Neural network layer wrapping DeltaThetaGammaNetwork dynamics.

    Provides a differentiable ``nn.Module`` interface to the 3-frequency
    hierarchical oscillator dynamics. Maps input features to oscillator
    states, propagates hierarchical dynamics, and returns concatenated
    amplitudes from all three bands.

    Args:
        n_delta: Number of Delta-band oscillators.
        n_theta: Number of Theta-band oscillators.
        n_gamma: Number of Gamma-band oscillators.
        n_dims: Input feature dimension.
        n_steps: Number of outer integration steps.
        dt: Outer timestep.
        coupling_strength: Intra-band coupling K.
        pac_depth: PAC modulation depth (used for both PAC links).

    Example:
        >>> layer = HierarchicalResonanceLayer(8, 16, 64, n_dims=128)
        >>> x = torch.randn(32, 128)
        >>> out = layer(x)
        >>> print(out.shape)
        torch.Size([32, 88])
    """

    def __init__(
        self,
        n_delta: int = 8,
        n_theta: int = 16,
        n_gamma: int = 64,
        n_dims: int = 256,
        n_steps: int = 10,
        dt: float = 0.01,
        coupling_strength: float = 2.0,
        pac_depth: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_delta = n_delta
        self.n_theta = n_theta
        self.n_gamma = n_gamma
        self.n_total = n_delta + n_theta + n_gamma
        self.n_dims = n_dims
        self.n_steps = n_steps
        self.dt = dt

        # Learnable input projection
        self.proj_delta = nn.Linear(n_dims, n_delta, bias=False)
        self.proj_theta = nn.Linear(n_dims, n_theta, bias=False)
        self.proj_gamma = nn.Linear(n_dims, n_gamma, bias=False)

        # Learnable PAC depths
        self.pac_depth_dt = nn.Parameter(torch.tensor(pac_depth))
        self.pac_depth_tg = nn.Parameter(torch.tensor(pac_depth))

        self._coupling_strength = coupling_strength

        # Initialize projections
        nn.init.xavier_uniform_(self.proj_delta.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_theta.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_gamma.weight, gain=0.5)

    def forward(
        self,
        x: Tensor,
        return_phase: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass through hierarchical resonance dynamics.

        Args:
            x: Input tensor ``(B, D)`` or ``(D,)``.
            return_phase: If ``True``, also return the concatenated
                final oscillatory phases alongside the amplitudes.

        Returns:
            Concatenated amplitudes ``(B, n_total)``; or, when
            *return_phase* is ``True``, a tuple
            ``(amplitudes, phases)`` where *phases* has the same shape
            as *amplitudes* and contains the final per-oscillator phases
            (wrapped to ``[-π, π]``).
        """
        from prinet.core.propagation import (
            DeltaThetaGammaNetwork,
            OscillatorState,
            PhaseAmplitudeCoupling,
        )

        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        B = x.shape[0]

        # Project input to per-band initial amplitudes
        delta_amp = torch.clamp(torch.abs(self.proj_delta(x)), min=_EPS)
        theta_amp = torch.clamp(torch.abs(self.proj_theta(x)), min=_EPS)
        gamma_amp = torch.clamp(torch.abs(self.proj_gamma(x)), min=_EPS)

        # Initial phases from input projection angles
        delta_phase = _wrap_phase(self.proj_delta(x))
        theta_phase = _wrap_phase(self.proj_theta(x))
        gamma_phase = _wrap_phase(self.proj_gamma(x))

        # Initial frequencies (fixed per band)
        delta_freq = torch.full(
            (B, self.n_delta), 2.0, device=x.device, dtype=x.dtype
        )
        theta_freq = torch.full(
            (B, self.n_theta), 6.0, device=x.device, dtype=x.dtype
        )
        gamma_freq = torch.full(
            (B, self.n_gamma), 40.0, device=x.device, dtype=x.dtype
        )

        # Create per-sample PAC with clamped learnable depths
        pac_dt = PhaseAmplitudeCoupling(
            modulation_depth=torch.clamp(
                self.pac_depth_dt, 0.0, 1.0
            ).item()
        )
        pac_tg = PhaseAmplitudeCoupling(
            modulation_depth=torch.clamp(
                self.pac_depth_tg, 0.0, 1.0
            ).item()
        )

        # Run dynamics per-sample (batched oscillator models don't
        # support heterogeneous initial states across batch)
        all_amps = []
        all_phases = []
        for b in range(B):
            ds = OscillatorState(
                phase=delta_phase[b], amplitude=delta_amp[b],
                frequency=delta_freq[b],
            )
            ts = OscillatorState(
                phase=theta_phase[b], amplitude=theta_amp[b],
                frequency=theta_freq[b],
            )
            gs = OscillatorState(
                phase=gamma_phase[b], amplitude=gamma_amp[b],
                frequency=gamma_freq[b],
            )

            net = DeltaThetaGammaNetwork(
                n_delta=self.n_delta,
                n_theta=self.n_theta,
                n_gamma=self.n_gamma,
                coupling_strength=self._coupling_strength,
                pac_depth_dt=pac_dt.modulation_depth,
                pac_depth_tg=pac_tg.modulation_depth,
                device=x.device,
                dtype=x.dtype,
            )
            final, _ = net.integrate(
                (ds, ts, gs), n_steps=self.n_steps, dt=self.dt
            )
            # Concatenate final amplitudes from all bands
            combined = torch.cat([
                final[0].amplitude,
                final[1].amplitude,
                final[2].amplitude,
            ])
            all_amps.append(combined)
            # Concatenate final phases from all bands (wrapped to [-π, π])
            combined_phase = _wrap_phase(torch.cat([
                final[0].phase,
                final[1].phase,
                final[2].phase,
            ]))
            all_phases.append(combined_phase)

        out = torch.stack(all_amps, dim=0)  # (B, n_total)
        phases = torch.stack(all_phases, dim=0)  # (B, n_total)

        if was_1d:
            out = out.squeeze(0)
            phases = phases.squeeze(0)

        if return_phase:
            return out, phases
        return out


class PhaseAmplitudeCouplingLayer(nn.Module):
    """Learnable phase-amplitude coupling as an ``nn.Module``.

    Wraps ``PhaseAmplitudeCoupling`` with a learnable modulation depth
    parameter, suitable for end-to-end training.

    Args:
        initial_depth: Initial modulation depth.

    Example:
        >>> layer = PhaseAmplitudeCouplingLayer(initial_depth=0.3)
        >>> slow_phase = torch.zeros(32, 16)
        >>> fast_amp = torch.ones(32, 64)
        >>> out = layer(slow_phase, fast_amp)
    """

    def __init__(self, initial_depth: float = 0.3) -> None:
        super().__init__()
        self.modulation_depth = nn.Parameter(
            torch.tensor(initial_depth)
        )

    def forward(
        self, slow_phase: Tensor, fast_amplitude: Tensor
    ) -> Tensor:
        """Apply learnable PAC modulation.

        Args:
            slow_phase: Slow-band phases ``(..., N_slow)``.
            fast_amplitude: Fast-band amplitudes ``(..., N_fast)``.

        Returns:
            Modulated amplitudes.
        """
        from prinet.core.propagation import PhaseAmplitudeCoupling

        depth = torch.clamp(self.modulation_depth, 0.0, 1.0).item()
        pac = PhaseAmplitudeCoupling(modulation_depth=depth)
        return pac.modulate(slow_phase, fast_amplitude)


# ---- Q3 Late: DG Layer & Phase-to-Rate Autoencoder ---------------------


class DGLayer(nn.Module):
    """Dentate Gyrus layer wrapper as a training-ready ``nn.Module``.

    Wraps :class:`prinet.core.propagation.DentateGyrusConverter` with
    learnable FFI/FBI parameters and proper gradient flow.

    Args:
        n_input: Number of input oscillators.
        top_k: Number of winners in FBI competition.
        ffi_delay: FFI delay in integration steps.
        fbi_delay: FBI delay in integration steps.
        n_integration_steps: EMA integration steps per forward pass.

    Example:
        >>> dg = DGLayer(n_input=64, top_k=8)
        >>> phase = torch.rand(32, 64)
        >>> amp = torch.ones(32, 64)
        >>> sparse_rates = dg(phase, amp)
        >>> print(sparse_rates.shape)
        torch.Size([32, 64])
    """

    def __init__(
        self,
        n_input: int,
        top_k: int = 8,
        ffi_delay: int = 2,
        fbi_delay: int = 20,
        n_integration_steps: int = 5,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.top_k = top_k
        self.n_integration_steps = n_integration_steps
        # Learnable parameters for modulating DG behavior
        self.ffi_scale = nn.Parameter(torch.tensor(1.0))
        self.fbi_temperature = nn.Parameter(torch.tensor(1.0))

        from prinet.core.propagation import DentateGyrusConverter

        self._converter = DentateGyrusConverter(
            n_oscillators=n_input,
            k=top_k,
            ffi_delay=ffi_delay,
            fbi_delay=fbi_delay,
        )

    def forward(self, phase: Tensor, amplitude: Tensor) -> Tensor:
        """Convert phase-amplitude → sparse rate codes.

        Args:
            phase: Phase tensor ``(B, N)`` or ``(N,)``.
            amplitude: Amplitude tensor, same shape.

        Returns:
            Sparse rate-coded output, same shape.
        """
        was_1d = phase.dim() == 1
        if was_1d:
            phase = phase.unsqueeze(0)
            amplitude = amplitude.unsqueeze(0)

        # Scale amplitude by learnable FFI modulation
        scaled_amp = amplitude * torch.clamp(self.ffi_scale, min=0.01)

        # Run DG conversion per batch element
        results: list[Tensor] = []
        for b in range(phase.shape[0]):
            sparse_rate = self._converter.convert(
                phase[b], scaled_amp[b], self.n_integration_steps
            )
            results.append(sparse_rate)

        out = torch.stack(results, dim=0)

        if was_1d:
            return out.squeeze(0)
        return out


class PhaseToRateAutoencoder(nn.Module):
    """Autoencoder with Phase-to-Rate bottleneck.

    Architecture: Encoder (dense → phase/amp) → PhaseToRate (sparse)
    → Decoder (rate → reconstruction).

    Measures information preservation through the sparse bottleneck.
    Useful for validating that phase-to-rate conversion retains
    discriminative information.

    Args:
        n_input: Input dimension (e.g. 784 for MNIST).
        n_oscillators: Number of oscillators in bottleneck.
        sparsity: Target sparsity for rate bottleneck.
        mode: WTA mode for PhaseToRateConverter.

    Example:
        >>> ae = PhaseToRateAutoencoder(784, 64, sparsity=0.1)
        >>> x = torch.randn(32, 784)
        >>> recon, rates = ae(x)
        >>> print(recon.shape, rates.shape)
        torch.Size([32, 784]) torch.Size([32, 64])
    """

    def __init__(
        self,
        n_input: int = 784,
        n_oscillators: int = 64,
        sparsity: float = 0.1,
        mode: str = "soft",
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_oscillators = n_oscillators

        # Encoder: input → phase + amplitude
        self.encoder_phase = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, n_oscillators),
        )
        self.encoder_amp = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, n_oscillators),
            nn.Softplus(),  # amplitudes > 0
        )

        # Phase-to-Rate bottleneck
        self.converter = PhaseToRateConverter(
            n_oscillators=n_oscillators,
            mode=mode,
            sparsity=sparsity,
        )

        # Decoder: sparse rates → reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(n_oscillators, 256),
            nn.ReLU(),
            nn.Linear(256, n_input),
        )

        # Classification head on bottleneck codes
        self.classifier = nn.Linear(n_oscillators, 10)

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through autoencoder.

        Args:
            x: Input tensor ``(B, D_input)``.

        Returns:
            Tuple of ``(reconstruction, sparse_rates)``.
        """
        phase = self.encoder_phase(x)
        amplitude = self.encoder_amp(x)
        sparse_rates = self.converter(phase, amplitude)
        reconstruction = self.decoder(sparse_rates)
        return reconstruction, sparse_rates

    def classify(self, x: Tensor) -> Tensor:
        """Classify using bottleneck codes.

        Args:
            x: Input tensor ``(B, D_input)``.

        Returns:
            Log-probabilities ``(B, 10)``.
        """
        _, rates = self.forward(x)
        return F.log_softmax(self.classifier(rates), dim=-1)


class DenseAutoencoder(nn.Module):
    """Dense autoencoder baseline (no sparsity bottleneck).

    Same capacity as ``PhaseToRateAutoencoder`` but without the
    phase-to-rate conversion or sparsity constraint.

    Args:
        n_input: Input dimension.
        n_bottleneck: Bottleneck dimension.
    """

    def __init__(
        self,
        n_input: int = 784,
        n_bottleneck: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, n_bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_bottleneck, 256),
            nn.ReLU(),
            nn.Linear(256, n_input),
        )
        self.classifier = nn.Linear(n_bottleneck, 10)

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Returns:
            Tuple of ``(reconstruction, bottleneck_codes)``.
        """
        codes = self.encoder(x)
        recon = self.decoder(codes)
        return recon, codes

    def classify(self, x: Tensor) -> Tensor:
        """Classify using bottleneck codes."""
        _, codes = self.forward(x)
        return F.log_softmax(self.classifier(codes), dim=-1)


# =========================================================================
# Year 2 Q1 — Workstream A: Discrete Multi-Rate Layer Wrapper
# =========================================================================


class DiscreteDeltaThetaGammaLayer(nn.Module):
    """Neural network layer wrapping :class:`DiscreteDeltaThetaGamma`.

    Drop-in replacement for :class:`HierarchicalResonanceLayer` using
    discrete-time dynamics instead of continuous ODE integration.
    All operations are batched — no per-sample loop.

    Maps input features ``(B, D)`` to oscillator amplitudes ``(B, n_total)``
    via:

    1. Learned projection (input → per-band initial phase & amplitude).
    2. ``n_steps`` discrete macro steps (each is a single matrix op,
       not an ODE sub-stepping procedure).
    3. Concatenated amplitudes as output.

    Forward cost: O(n_steps × n_total²) — typically 10× fewer FLOPs than
    :class:`HierarchicalResonanceLayer` for the same ``n_steps``.

    Args:
        n_delta: Number of Delta-band oscillators.
        n_theta: Number of Theta-band oscillators.
        n_gamma: Number of Gamma-band oscillators.
        n_dims: Input feature dimension.
        n_steps: Number of discrete macro steps.
        dt: Timestep per macro step.
        coupling_strength: Initial intra-band coupling magnitude.
        pac_depth: Initial PAC modulation depth.

    Example:
        >>> layer = DiscreteDeltaThetaGammaLayer(4, 8, 32, n_dims=128)
        >>> x = torch.randn(16, 128)
        >>> out = layer(x)
        >>> print(out.shape)
        torch.Size([16, 44])
    """

    def __init__(
        self,
        n_delta: int = 8,
        n_theta: int = 16,
        n_gamma: int = 64,
        n_dims: int = 256,
        n_steps: int = 10,
        dt: float = 0.01,
        coupling_strength: float = 2.0,
        pac_depth: float = 0.3,
    ) -> None:
        super().__init__()
        from prinet.core.propagation import DiscreteDeltaThetaGamma

        self.n_delta = n_delta
        self.n_theta = n_theta
        self.n_gamma = n_gamma
        self.n_total = n_delta + n_theta + n_gamma
        self.n_dims = n_dims
        self.n_steps = n_steps
        self.dt = dt

        # Input → per-band phase & amplitude projections
        self.proj_phase = nn.Linear(n_dims, self.n_total, bias=False)
        self.proj_amplitude = nn.Linear(n_dims, self.n_total, bias=False)

        nn.init.xavier_uniform_(self.proj_phase.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_amplitude.weight, gain=0.5)

        # Discrete dynamics engine (nn.Module — parameters registered)
        self.dynamics = DiscreteDeltaThetaGamma(
            n_delta=n_delta,
            n_theta=n_theta,
            n_gamma=n_gamma,
            coupling_strength=coupling_strength,
            pac_depth=pac_depth,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through discrete hierarchical dynamics.

        Args:
            x: Input tensor ``(B, D)`` or ``(D,)``.

        Returns:
            Concatenated amplitudes ``(B, n_total)`` or ``(n_total,)``.
        """
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        # Project to initial phases and amplitudes
        phase_0 = _wrap_phase(self.proj_phase(x))  # (B, n_total)
        amp_0 = torch.clamp(
            torch.abs(self.proj_amplitude(x)), min=_EPS
        )  # (B, n_total)

        # Run discrete integration
        _final_phase, final_amp = self.dynamics.integrate(
            phase_0, amp_0, n_steps=self.n_steps, dt=self.dt
        )

        if was_1d:
            return final_amp.squeeze(0)
        return final_amp


# =========================================================================
# Year 2 Q1 — Workstream B: Oscillatory Attention
# =========================================================================


class OscillatoryAttention(nn.Module):
    """Multi-head attention with additive oscillatory coherence bias.

    Extends standard scaled dot-product attention by adding a learnable
    phase-coherence bias to the attention scores:

        ``score = (Q @ K^T) / √d_k + α · coherence(φ_i, φ_j)``

    where ``coherence(φ_i, φ_j) = cos(φ_i − φ_j)`` computed from
    per-token phase representations, and ``α`` is a learnable scalar
    per head.

    This biases the attention pattern toward tokens whose oscillatory
    phases are aligned (i.e., bound together), implementing a soft
    form of oscillatory binding in the attention mechanism.

    The layer expects input of shape ``(B, S, D)`` where ``S`` is the
    sequence length (number of tokens/oscillators) and ``D`` is the
    model dimension. Phase information can be:

    - **Derived from input** via a learned projection (default).
    - **Provided externally** via the ``phase`` argument.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout rate.

    Example:
        >>> attn = OscillatoryAttention(d_model=64, n_heads=4)
        >>> x = torch.randn(8, 10, 64)
        >>> out = attn(x)
        >>> print(out.shape)
        torch.Size([8, 10, 64])
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Standard QKV projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Phase projection: input → per-token phase (1 scalar per head)
        self.phase_proj = nn.Linear(d_model, n_heads)

        # Learnable coherence bias strength per head
        self.alpha = nn.Parameter(torch.zeros(n_heads))

        self.dropout = nn.Dropout(dropout)
        self._scale = self.d_k ** -0.5

    def forward(
        self,
        x: Tensor,
        phase: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with oscillatory coherence bias.

        Args:
            x: Input tensor ``(B, S, D)``.
            phase: Optional external phase ``(B, S, n_heads)``.
                If ``None``, phases are derived from ``x``.
            mask: Optional attention mask ``(B, S, S)`` or ``(S, S)``.

        Returns:
            Output tensor ``(B, S, D)``.
        """
        B, S, D = x.shape

        # QKV projections → (B, n_heads, S, d_k)
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        # Standard attention scores: (B, n_heads, S, S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self._scale

        # Compute phase coherence bias
        if phase is None:
            phase = self.phase_proj(x)  # (B, S, n_heads)

        # phase: (B, S, n_heads) → (B, n_heads, S)
        phase_h = phase.transpose(1, 2)

        # Coherence: cos(φ_i - φ_j) → (B, n_heads, S, S)
        phase_i = phase_h.unsqueeze(-1)  # (B, n_heads, S, 1)
        phase_j = phase_h.unsqueeze(-2)  # (B, n_heads, 1, S)
        coherence = torch.cos(phase_i - phase_j)  # (B, n_heads, S, S)

        # Apply learnable alpha per head: (n_heads,) → (1, n_heads, 1, 1)
        alpha = self.alpha.view(1, self.n_heads, 1, 1)
        scores = scores + alpha * coherence

        # Optional mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax + dropout → attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, V)  # (B, n_heads, S, d_k)
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(B, S, D)
        )

        out: Tensor = self.W_o(attn_out)
        return out
