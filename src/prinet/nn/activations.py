"""Oscillator-Compatible Activation Functions for PRINet.

Provides activation functions tailored for oscillatory neural networks,
including the derivative-of-SiLU (dSiLU) and holomorphic-compatible
versions that preserve analyticity for complex-valued forward passes.

The key activation is ``dSiLU(z) = σ(z) · (1 + z · (1 − σ(z)))``
where σ is the logistic sigmoid. This is the exact derivative of SiLU
(Swish) and has several desirable properties for oscillatory networks:

* Smooth and bounded-output for phase-like inputs.
* Non-monotonic — captures interference patterns.
* Naturally normalised: ``dSiLU(0) = 0.5``.

Reference:
    PRINet_Year1_Q2_Plan.md — P2 Activations task.

Example:
    >>> import torch
    >>> from prinet.nn.activations import dSiLU
    >>> act = dSiLU()
    >>> x = torch.randn(32, 128)
    >>> y = act(x)
    >>> print(y.shape)
    torch.Size([32, 128])
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class dSiLU(nn.Module):
    """Derivative of SiLU (Swish) activation.

    Computes ``σ(z) · (1 + z · (1 − σ(z)))`` which is the exact
    analytical derivative of ``SiLU(z) = z · σ(z)``.

    This activation is non-monotonic and naturally bounded, making
    it well-suited for phase-coupled oscillatory layers where
    unbounded activations can lead to NaN divergence.

    The numerically-stable formulation avoids computing
    ``z * (1 - σ(z))`` as a product by rewriting as
    ``σ(z) + z · σ(z) · (1 − σ(z))``.

    Example:
        >>> act = dSiLU()
        >>> x = torch.linspace(-5, 5, 100)
        >>> y = act(x)
        >>> assert y.min() > -0.3  # bounded below
        >>> assert y.max() < 1.2   # bounded above
    """

    def forward(self, z: Tensor) -> Tensor:
        """Apply dSiLU activation.

        Args:
            z: Input tensor of any shape.

        Returns:
            Activated tensor of the same shape.
        """
        sig = torch.sigmoid(z)
        return sig + z * sig * (1.0 - sig)


class HolomorphicActivation(nn.Module):
    """Holomorphic-compatible activation for complex oscillator states.

    Applies a smooth, bounded activation that is well-behaved for
    complex-valued inputs. When the input is real, this reduces to a
    scaled ``tanh``. For complex inputs, the real and imaginary parts
    are activated independently (split-complex approach) to maintain
    compatibility with backpropagation.

    If ``holomorphic=True``, the activation uses the complex ``tanh``
    extension ``tanh(z)`` which is holomorphic everywhere except at
    poles ``z = i(π/2 + kπ)``.

    Args:
        scale: Output scaling factor.
        holomorphic: If ``True``, use complex tanh. If ``False``,
            apply real tanh to real and imaginary parts independently.

    Example:
        >>> act = HolomorphicActivation(holomorphic=True)
        >>> z = torch.randn(32, 64, dtype=torch.complex64)
        >>> y = act(z)
        >>> assert y.dtype == torch.complex64
    """

    def __init__(
        self, scale: float = 1.0, holomorphic: bool = False
    ) -> None:
        super().__init__()
        self._scale = scale
        self._holomorphic = holomorphic

    def forward(self, z: Tensor) -> Tensor:
        """Apply holomorphic activation.

        Args:
            z: Input tensor (real or complex).

        Returns:
            Activated tensor of the same dtype and shape.
        """
        if self._holomorphic and z.is_complex():
            return self._scale * torch.tanh(z)
        elif z.is_complex():
            return self._scale * torch.complex(
                torch.tanh(z.real), torch.tanh(z.imag)
            )
        else:
            return self._scale * torch.tanh(z)


class PhaseActivation(nn.Module):
    """Phase-aware activation that wraps output to [0, 2π).

    Useful as a final activation for layers that output phase values.
    Applies a smooth nonlinearity followed by modular wrapping.

    Args:
        activation: Inner activation to apply before wrapping.
            Defaults to ``dSiLU`` if ``None``.

    Example:
        >>> act = PhaseActivation()
        >>> x = torch.randn(32, 64)
        >>> y = act(x)
        >>> assert (y >= 0).all() and (y < 2 * torch.pi).all()
    """

    def __init__(self, activation: Optional[nn.Module] = None) -> None:
        super().__init__()
        self._inner = activation or dSiLU()

    def forward(self, z: Tensor) -> Tensor:
        """Apply phase activation with wrapping.

        Args:
            z: Input tensor.

        Returns:
            Phase-wrapped activated tensor in [0, 2π).
        """
        TWO_PI = 2.0 * torch.pi
        wrapped = self._inner(z) % TWO_PI
        # Clamp to handle floating-point edge case where remainder == 2π
        return torch.clamp(wrapped, min=0.0, max=TWO_PI - 1e-7)


class GatedPhaseActivation(nn.Module):
    """Phase activation with a learnable gating mechanism.

    Extends ``PhaseActivation`` with a sigmoid-based gate that
    controls how much of the phase information passes through:

        y = gate(z) · PhaseActivation(z)

    where ``gate(z) = σ(W_g z + b_g)`` is a per-unit learnable gate.

    Args:
        n_dims: Dimension of the input (for per-unit gate parameters).
        activation: Inner activation before wrapping. Default ``dSiLU``.

    Example:
        >>> act = GatedPhaseActivation(64)
        >>> x = torch.randn(32, 64)
        >>> y = act(x)
        >>> assert (y >= 0).all() and (y < 2 * torch.pi).all()
    """

    def __init__(
        self,
        n_dims: int,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self._phase_act = PhaseActivation(activation)
        self.gate_weight = nn.Parameter(torch.zeros(n_dims))
        self.gate_bias = nn.Parameter(torch.zeros(n_dims))

    def forward(self, z: Tensor) -> Tensor:
        """Apply gated phase activation.

        Args:
            z: Input tensor ``(..., D)``.

        Returns:
            Gated phase-wrapped tensor in [0, 2π).
        """
        gate = torch.sigmoid(self.gate_weight * z + self.gate_bias)
        phase_out = self._phase_act(z)
        result: Tensor = gate * phase_out
        return result
