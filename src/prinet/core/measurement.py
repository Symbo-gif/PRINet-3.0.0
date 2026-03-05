"""Synchronization Measurement Metrics for PRINet.

Provides functions for computing the Kuramoto order parameter,
phase coherence, and spectral analysis of oscillator states. These
metrics are critical for monitoring training stability and detecting
desynchronization catastrophes.

Sparse-aware measurement functions (``sparse_mean_phase_coherence``,
``sparse_synchronization_energy``) use pre-computed k-NN neighbour
indices to provide O(N·k) approximations of the O(N²) full metrics.
Use :func:`prinet.core.propagation._build_phase_knn_index` or
:func:`build_phase_knn` to obtain the index tensor.

Example:
    >>> import torch
    >>> from prinet.core.propagation import OscillatorState
    >>> state = OscillatorState.create_random(100, seed=42)
    >>> r = kuramoto_order_parameter(state.phase)
    >>> print(f"Order parameter: {r:.4f}")
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def kuramoto_order_parameter(phase: Tensor) -> Tensor:
    """Compute the Kuramoto order parameter r(t).

    The order parameter measures the degree of phase synchronization:

        r = |1/N Σᵢ exp(i φᵢ)|

    where r=0 indicates incoherence and r=1 indicates full synchronization.

    Args:
        phase: Phase tensor of shape ``(N,)`` or ``(B, N)`` where B is
            the batch dimension.

    Returns:
        Scalar order parameter if unbatched, or tensor of shape ``(B,)``
        for batched input. Values in [0, 1].

    Raises:
        ValueError: If phase tensor is empty.

    Example:
        >>> phases = torch.zeros(100)  # All synchronized
        >>> r = kuramoto_order_parameter(phases)
        >>> assert torch.isclose(r, torch.tensor(1.0), atol=1e-6)
    """
    if phase.numel() == 0:
        raise ValueError("Phase tensor must not be empty.")

    # Complex exponential: exp(iφ)
    z = torch.exp(1j * phase.to(torch.float64))

    # Mean over oscillator dimension (last axis)
    mean_z = z.mean(dim=-1)

    # Order parameter is the magnitude
    r = torch.abs(mean_z).to(phase.dtype)
    return r


def kuramoto_order_parameter_complex(phase: Tensor) -> Tensor:
    """Compute the complex Kuramoto order parameter.

    Returns the full complex mean-field:

        Z = 1/N Σᵢ exp(i φᵢ) = r exp(iψ)

    where r is the synchronization magnitude and ψ is the mean phase.

    Args:
        phase: Phase tensor of shape ``(N,)`` or ``(B, N)``.

    Returns:
        Complex tensor representing Z. Shape ``()`` or ``(B,)``.

    Raises:
        ValueError: If phase tensor is empty.
    """
    if phase.numel() == 0:
        raise ValueError("Phase tensor must not be empty.")
    z = torch.exp(1j * phase.to(torch.float64))
    return z.mean(dim=-1)


def mean_phase_coherence(phase: Tensor) -> Tensor:
    """Compute mean pairwise phase coherence.

    Measures average coherence across all oscillator pairs:

        C = (2 / N(N-1)) Σᵢ<ⱼ cos(φᵢ - φⱼ)

    Values near 1 indicate high synchronization; values near 0 indicate
    incoherence.

    Args:
        phase: Phase tensor of shape ``(N,)`` or ``(B, N)``.

    Returns:
        Scalar coherence if unbatched, or ``(B,)`` for batched input.
        Values in ``[-1, 1]``.

    Raises:
        ValueError: If fewer than 2 oscillators are provided.

    Example:
        >>> phases = torch.tensor([0.0, 0.1, -0.1, 0.05])
        >>> c = mean_phase_coherence(phases)
        >>> assert c > 0.9  # Nearly synchronized
    """
    n = phase.shape[-1]
    if n < 2:
        raise ValueError(
            f"Need at least 2 oscillators for coherence, got {n}."
        )

    # Phase difference matrix: cos(φᵢ - φⱼ)
    phase_diff = phase.unsqueeze(-1) - phase.unsqueeze(-2)  # (..., N, N)
    cos_diff = torch.cos(phase_diff)

    # Sum upper triangle (excluding diagonal)
    # mask[i, j] = 1 if i < j
    mask = torch.triu(
        torch.ones(n, n, device=phase.device, dtype=phase.dtype),
        diagonal=1,
    )
    n_pairs = n * (n - 1) / 2.0
    coherence = (cos_diff * mask).sum(dim=(-2, -1)) / n_pairs
    return coherence


def phase_coherence_matrix(phase: Tensor) -> Tensor:
    """Compute the full pairwise phase coherence matrix.

    Returns ``C[i,j] = cos(φᵢ - φⱼ)``, which can be used to identify
    clusters of synchronized oscillators.

    Args:
        phase: Phase tensor of shape ``(N,)`` or ``(B, N)``.

    Returns:
        Coherence matrix of shape ``(N, N)`` or ``(B, N, N)``.
    """
    phase_diff = phase.unsqueeze(-1) - phase.unsqueeze(-2)
    return torch.cos(phase_diff)


def power_spectral_density(
    amplitude: Tensor,
    phase: Tensor,
    n_freq_bins: Optional[int] = None,
) -> Tensor:
    """Compute the power spectral density of the resonance state.

    Constructs the complex signal r(t) = Σᵢ rᵢ exp(iφᵢ) and computes
    its discrete Fourier transform power spectrum: P(f) = |R̂(f)|².

    Args:
        amplitude: Amplitude tensor of shape ``(N,)`` or ``(B, N)``.
        phase: Phase tensor of same shape as amplitude.
        n_freq_bins: Number of FFT bins. Defaults to ``2*N``.

    Returns:
        Power spectrum of shape ``(n_freq_bins,)`` or
        ``(B, n_freq_bins)``.
    """
    n = amplitude.shape[-1]
    if n_freq_bins is None:
        n_freq_bins = 2 * n

    # Complex resonance signal
    signal = amplitude * torch.exp(
        1j * phase.to(torch.float64)
    ).to(torch.complex64)

    # FFT and power spectrum
    spectrum = torch.fft.fft(signal, n=n_freq_bins, dim=-1)
    power = torch.abs(spectrum) ** 2
    return power.to(amplitude.dtype)


def extract_concept_probabilities(
    amplitude: Tensor,
    phase: Tensor,
    concept_frequencies: Tensor,
    concept_bandwidths: Tensor,
    n_freq_bins: Optional[int] = None,
) -> Tensor:
    """Extract concept probabilities from the resonance spectrum.

    For each concept k with center frequency ωₖ and bandwidth Δωₖ,
    computes:

        P(k) ∝ Σ_{f ∈ band_k} P(f)

    where P(f) is the power spectral density.

    Args:
        amplitude: Amplitude tensor of shape ``(N,)`` or ``(B, N)``.
        phase: Phase tensor of same shape.
        concept_frequencies: Center frequencies of shape ``(K,)``.
        concept_bandwidths: Bandwidths of shape ``(K,)``.
        n_freq_bins: Number of FFT bins. Defaults to ``2*N``.

    Returns:
        Probability distribution of shape ``(K,)`` or ``(B, K)``.
    """
    n = amplitude.shape[-1]
    if n_freq_bins is None:
        n_freq_bins = 2 * n

    power = power_spectral_density(amplitude, phase, n_freq_bins)
    freq_indices = torch.arange(
        n_freq_bins, device=amplitude.device, dtype=amplitude.dtype
    )

    k = concept_frequencies.shape[0]
    batched = amplitude.dim() > 1

    if batched:
        probs = torch.zeros(
            amplitude.shape[0], k, device=amplitude.device, dtype=amplitude.dtype
        )
    else:
        probs = torch.zeros(
            k, device=amplitude.device, dtype=amplitude.dtype
        )

    for idx in range(k):
        band_mask = (
            torch.abs(freq_indices - concept_frequencies[idx])
            < concept_bandwidths[idx]
        )
        if batched:
            probs[:, idx] = power[:, band_mask].sum(dim=-1)
        else:
            probs[idx] = power[band_mask].sum()

    # Normalize to probability distribution
    total = probs.sum(dim=-1, keepdim=True)
    probs = probs / (total + 1e-10)
    return probs


def synchronization_energy(
    phase: Tensor,
    amplitude: Tensor,
    coupling_matrix: Optional[Tensor] = None,
) -> Tensor:
    """Compute the synchronization energy of the oscillator system.

    E(φ, r) = -Σᵢⱼ Kᵢⱼ cos(φᵢ - φⱼ) |rᵢ| |rⱼ|

    Lower energy indicates greater synchronization.

    Args:
        phase: Phase tensor of shape ``(N,)`` or ``(B, N)``.
        amplitude: Amplitude tensor of same shape.
        coupling_matrix: Optional coupling matrix ``(N, N)``.
            Defaults to uniform ``1/N``.

    Returns:
        Scalar energy or ``(B,)`` for batched input.
    """
    n = phase.shape[-1]
    if coupling_matrix is None:
        coupling_matrix = torch.full(
            (n, n),
            1.0 / n,
            device=phase.device,
            dtype=phase.dtype,
        )
        coupling_matrix.fill_diagonal_(0.0)

    phase_diff = phase.unsqueeze(-1) - phase.unsqueeze(-2)  # (..., N, N)
    cos_diff = torch.cos(phase_diff)

    amp_outer = amplitude.unsqueeze(-1) * amplitude.unsqueeze(-2)

    energy = -(coupling_matrix * cos_diff * amp_outer).sum(dim=(-2, -1))
    return energy


# =========================================================================
# Sparse-aware measurement functions — O(N·k) approximations
# =========================================================================

_SPARSE_MEAS_EPS: float = 1e-10
"""Epsilon guard for sparse measurement denominators."""


def build_phase_knn(
    phase: Tensor,
    k: int,
) -> Tensor:
    """Build k-nearest-phase-neighbour index for measurement use.

    Convenience wrapper around the propagation helper that accepts
    user-facing ``(N,)`` or ``(B, N)`` tensors and returns an index
    suitable for :func:`sparse_mean_phase_coherence` and
    :func:`sparse_synchronization_energy`.

    Args:
        phase: Phase tensor of shape ``(N,)`` or ``(B, N)``.
        k: Number of nearest neighbours.

    Returns:
        Neighbour index tensor of shape ``(B, N, k)`` (or ``(1, N, k)``
        for unbatched input).

    Raises:
        ValueError: If ``k < 1`` or ``k >= N``.

    Example:
        >>> phases = torch.linspace(0, 2 * 3.14159, 20)
        >>> nbr = build_phase_knn(phases, k=4)
        >>> print(nbr.shape)
        torch.Size([1, 20, 4])
    """
    from prinet.core.propagation import _build_phase_knn_index

    n = phase.shape[-1]
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if k >= n:
        raise ValueError(
            f"k must be < N ({n}), got {k}. Use dense metrics instead."
        )

    flat = phase.reshape(-1, n)  # (B, N)
    return _build_phase_knn_index(flat, k)


def sparse_mean_phase_coherence(
    phase: Tensor,
    nbr_idx: Tensor,
) -> Tensor:
    """Sparse mean phase coherence — O(N·k) approximation.

    For each oscillator *i*, measures the local coherence over its
    *k* nearest phase neighbours:

        c_i = (1/k) |Σ_{j ∈ kNN(i)} exp(i(φ_j − φ_i))|

    The global coherence is the mean of these local values.

    This approximates :func:`mean_phase_coherence` but avoids the
    O(N²) pairwise difference matrix.  For k ≈ log₂ N the cost is
    O(N log N).

    Args:
        phase: Phase tensor of shape ``(N,)`` or ``(B, N)``.
        nbr_idx: Neighbour index tensor of shape ``(B, N, k)`` as
            returned by :func:`build_phase_knn`.

    Returns:
        Scalar coherence (or ``(B,)`` for batched input) in ``[0, 1]``.

    Raises:
        ValueError: If phase tensor has fewer than 2 oscillators.

    Example:
        >>> phases = torch.zeros(100)  # All same → coherence ≈ 1
        >>> nbr = build_phase_knn(phases, k=8)
        >>> c = sparse_mean_phase_coherence(phases, nbr)
        >>> assert c > 0.99
    """
    n = phase.shape[-1]
    if n < 2:
        raise ValueError(
            f"Need at least 2 oscillators for coherence, got {n}."
        )

    was_1d = phase.dim() == 1
    flat_phase = phase.reshape(-1, n)  # (B, N)
    B = flat_phase.shape[0]
    k = nbr_idx.shape[-1]

    # Gather neighbour phases → (B, N, k)
    nbr_phase = flat_phase.gather(
        1, nbr_idx.reshape(B, -1)
    ).reshape(B, n, k)

    # Wrapped phase differences
    phase_i = flat_phase.unsqueeze(-1)  # (B, N, 1)
    delta = nbr_phase - phase_i
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))

    # Local coherence: |mean exp(iΔ)| per oscillator
    z = torch.exp(1j * delta.to(torch.float64))  # (B, N, k)
    local_r = torch.abs(z.mean(dim=-1)).to(phase.dtype)  # (B, N)

    # Global coherence
    coherence = local_r.mean(dim=-1)  # (B,)

    if was_1d:
        return coherence.squeeze(0)
    return coherence


def inter_frame_phase_correlation(
    phase_t: Tensor,
    phase_t_prev: Tensor,
) -> Tensor:
    """Compute circular correlation between phases at consecutive frames.

    Measures how well phase assignments are preserved across time steps,
    which is critical for temporal binding:

        ρ = |⟨exp(i(φ_t − φ_{t-1}))⟩|

    A value near 1 indicates strong phase continuity (same binding
    pattern preserved); a value near 0 indicates phase reassignment.

    This is the circular analogue of Pearson correlation — it measures
    the consistency of phase *differences* across the population.

    Args:
        phase_t: Phase tensor at time t, shape ``(N,)`` or ``(B, N)``.
        phase_t_prev: Phase tensor at time t-1, same shape.

    Returns:
        Scalar correlation (or ``(B,)`` for batched input) in ``[0, 1]``.

    Raises:
        ValueError: If tensors have different shapes or are empty.

    Example:
        >>> prev = torch.rand(100) * 2 * 3.14159
        >>> curr = prev + 0.01  # small phase advance → high correlation
        >>> rho = inter_frame_phase_correlation(curr, prev)
        >>> assert rho > 0.99
    """
    if phase_t.shape != phase_t_prev.shape:
        raise ValueError(
            f"Shape mismatch: phase_t {phase_t.shape} vs "
            f"phase_t_prev {phase_t_prev.shape}."
        )
    if phase_t.numel() == 0:
        raise ValueError("Phase tensors must not be empty.")

    # Circular difference: exp(i(φ_t − φ_{t-1}))
    delta = phase_t - phase_t_prev
    z = torch.exp(1j * delta.to(torch.float64))

    # Mean over oscillator dimension (last axis)
    mean_z = z.mean(dim=-1)

    # Correlation magnitude
    rho = torch.abs(mean_z).to(phase_t.dtype)
    return rho


def sparse_synchronization_energy(
    phase: Tensor,
    amplitude: Tensor,
    nbr_idx: Tensor,
    coupling_strength: float = 1.0,
) -> Tensor:
    """Sparse synchronization energy — O(N·k) approximation.

    Computes the coupling energy only over k-NN edges:

        E_sparse = -(K/k) Σ_i Σ_{j ∈ kNN(i)} cos(φ_i − φ_j) |r_i| |r_j|

    Normalised so that the per-oscillator energy magnitude is comparable
    to the dense :func:`synchronization_energy`.

    Args:
        phase: Phase tensor of shape ``(N,)`` or ``(B, N)``.
        amplitude: Amplitude tensor of same shape.
        nbr_idx: Neighbour index ``(B, N, k)``.
        coupling_strength: Global coupling ``K``.

    Returns:
        Scalar energy or ``(B,)`` for batched input.

    Example:
        >>> phases = torch.zeros(50)
        >>> amps = torch.ones(50)
        >>> nbr = build_phase_knn(phases, k=4)
        >>> e = sparse_synchronization_energy(phases, amps, nbr)
        >>> assert e.item() < 0  # Sync'd → low energy
    """
    n = phase.shape[-1]
    was_1d = phase.dim() == 1
    flat_phase = phase.reshape(-1, n)
    flat_amp = amplitude.reshape(-1, n)
    B = flat_phase.shape[0]
    k = nbr_idx.shape[-1]

    # Gather neighbour phases and amplitudes
    nbr_phase = flat_phase.gather(
        1, nbr_idx.reshape(B, -1)
    ).reshape(B, n, k)
    nbr_amp = flat_amp.gather(
        1, nbr_idx.reshape(B, -1)
    ).reshape(B, n, k)

    # Wrapped phase differences
    phase_i = flat_phase.unsqueeze(-1)
    delta = nbr_phase - phase_i
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))

    cos_diff = torch.cos(delta)  # (B, N, k)
    amp_i = flat_amp.unsqueeze(-1)  # (B, N, 1)

    # Per-edge energy: -(K/k) cos(Δ) |rᵢ| |rⱼ|
    K_eff = coupling_strength / max(k, 1)
    edge_energy = -K_eff * cos_diff * amp_i * nbr_amp  # (B, N, k)

    energy = edge_energy.sum(dim=(-2, -1))  # (B,)

    if was_1d:
        return energy.squeeze(0)
    return energy
