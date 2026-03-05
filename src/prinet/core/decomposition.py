"""Polyadic Tensor Decomposition for PRINet.

Provides Tucker (HOSVD) and CP decomposition of higher-order tensors,
used to extract resonance modes from input data. Decoupled from
oscillator dynamics per Task 1.2a.

Example:
    >>> import torch
    >>> tensor = PolyadicTensor(shape=(8, 8, 8), rank=4)
    >>> data = torch.randn(8, 8, 8)
    >>> tensor.decompose(data)
    >>> reconstructed = tensor.reconstruct()
    >>> print(reconstructed.shape)
    torch.Size([8, 8, 8])
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor


class DecompositionError(Exception):
    """Raised when tensor decomposition fails or produces invalid results."""

    pass


class DimensionsMismatchError(Exception):
    """Raised when tensor dimensions do not match expected shape."""

    pass


class TensorDecompositionBase(ABC):
    """Abstract base class for tensor decomposition methods.

    All decomposition methods must implement ``decompose`` and
    ``reconstruct`` to provide a consistent API.

    Args:
        shape: Expected shape of the input tensor.
        rank: Target rank for the decomposition.
    """

    def __init__(self, shape: Tuple[int, ...], rank: int) -> None:
        if rank < 1:
            raise ValueError(
                f"Rank must be a positive integer, got {rank}."
            )
        if any(s < 1 for s in shape):
            raise ValueError(
                f"All shape dimensions must be positive, got {shape}."
            )
        self._shape = shape
        self._rank = rank

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the tensor this decomposition targets."""
        return self._shape

    @property
    def rank(self) -> int:
        """Target rank of the decomposition."""
        return self._rank

    @abstractmethod
    def decompose(self, tensor: Tensor) -> None:
        """Decompose the given tensor in-place.

        Args:
            tensor: Input tensor whose shape must match ``self.shape``.

        Raises:
            DimensionsMismatchError: If ``tensor.shape != self.shape``.
        """
        ...

    @abstractmethod
    def reconstruct(self) -> Tensor:
        """Reconstruct the tensor from stored factors.

        Returns:
            Reconstructed tensor approximation.

        Raises:
            DecompositionError: If decomposition has not been performed yet.
        """
        ...


class PolyadicTensor(TensorDecompositionBase):
    """Tucker (HOSVD) tensor decomposition.

    Decomposes an N-way tensor ``X`` into a core tensor ``G`` and
    orthogonal factor matrices ``U^(n)``:

        X ≈ G ×₁ U^(1) ×₂ U^(2) ... ×_N U^(N)

    Args:
        shape: Shape of the input tensor ``(I₁, I₂, ..., I_N)``.
        rank: Number of singular vectors to retain per mode.
        device: Torch device for factor storage.
        dtype: Data type for factor matrices.

    Example:
        >>> pt = PolyadicTensor(shape=(16, 16, 16), rank=4)
        >>> data = torch.randn(16, 16, 16)
        >>> pt.decompose(data)
        >>> approx = pt.reconstruct()
        >>> error = torch.norm(data - approx) / torch.norm(data)
        >>> print(f"Relative error: {error:.4f}")
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        rank: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(shape, rank)
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._factors: List[Tensor] = []
        self._core: Optional[Tensor] = None
        self._decomposed = False

    @property
    def factors(self) -> List[Tensor]:
        """Orthogonal factor matrices from the decomposition."""
        if not self._decomposed:
            raise DecompositionError(
                "No decomposition performed yet. Call decompose() first."
            )
        return self._factors

    @property
    def core(self) -> Tensor:
        """Core tensor from the Tucker decomposition."""
        if self._core is None:
            raise DecompositionError(
                "No decomposition performed yet. Call decompose() first."
            )
        return self._core

    @staticmethod
    def _mode_n_unfold(tensor: Tensor, mode: int) -> Tensor:
        """Unfold a tensor along the given mode.

        Args:
            tensor: N-dimensional tensor.
            mode: Mode index along which to unfold (0-based).

        Returns:
            2D matrix of shape ``(I_mode, prod(I_other))``.
        """
        n_dims = tensor.dim()
        perm = [mode] + [i for i in range(n_dims) if i != mode]
        unfolded = tensor.permute(*perm).contiguous()
        return unfolded.reshape(tensor.shape[mode], -1)

    @staticmethod
    def _mode_n_product(tensor: Tensor, matrix: Tensor, mode: int) -> Tensor:
        """Compute the mode-n product of a tensor with a matrix.

        Args:
            tensor: N-dimensional tensor.
            matrix: 2D matrix of shape ``(J, I_mode)``.
            mode: Mode index along which to multiply (0-based).

        Returns:
            Tensor with ``shape[mode]`` replaced by ``J``.
        """
        n_dims = tensor.dim()
        perm = [mode] + [i for i in range(n_dims) if i != mode]
        unfolded = tensor.permute(*perm).contiguous()
        shape_list = list(unfolded.shape)
        unfolded_2d = unfolded.reshape(shape_list[0], -1)
        result_2d = matrix @ unfolded_2d
        shape_list[0] = matrix.shape[0]
        result = result_2d.reshape(shape_list)
        inv_perm = [0] * n_dims
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return result.permute(*inv_perm).contiguous()

    def decompose(self, tensor: Tensor) -> None:
        """Perform Higher-Order SVD (Tucker) decomposition.

        Computes factor matrices via truncated SVD on each mode
        unfolding, then derives the core tensor.

        Args:
            tensor: Input tensor whose shape must match ``self.shape``.

        Raises:
            DimensionsMismatchError: If shapes do not match.
        """
        if tuple(tensor.shape) != self._shape:
            raise DimensionsMismatchError(
                f"Expected shape {self._shape}, got {tuple(tensor.shape)}."
            )
        tensor = tensor.to(device=self._device, dtype=self._dtype)
        self._factors = []
        rank = min(self._rank, min(self._shape))

        for mode in range(tensor.dim()):
            unfolded = self._mode_n_unfold(tensor, mode)
            u, _, _ = torch.linalg.svd(unfolded, full_matrices=False)
            # Truncate to target rank
            cols = min(rank, u.shape[1])
            self._factors.append(u[:, :cols])

        # Compute core tensor: G = X ×₁ U₁ᵀ ×₂ U₂ᵀ ...
        core = tensor.clone()
        for mode, factor in enumerate(self._factors):
            core = self._mode_n_product(core, factor.T, mode)
        self._core = core
        self._decomposed = True

    def reconstruct(self) -> Tensor:
        """Reconstruct the tensor from core and factor matrices.

        Returns:
            Approximate reconstruction of the original tensor.

        Raises:
            DecompositionError: If ``decompose()`` has not been called.
        """
        if not self._decomposed or self._core is None:
            raise DecompositionError(
                "No decomposition performed yet. Call decompose() first."
            )
        result = self._core.clone()
        for mode, factor in enumerate(self._factors):
            result = self._mode_n_product(result, factor, mode)
        return result

    def reconstruction_error(self, original: Tensor) -> float:
        """Compute relative Frobenius-norm reconstruction error.

        Args:
            original: The original tensor to compare against.

        Returns:
            Relative error ``||X - X̂||_F / ||X||_F``.

        Raises:
            DimensionsMismatchError: If shape does not match.
            DecompositionError: If decomposition has not been performed.
        """
        if tuple(original.shape) != self._shape:
            raise DimensionsMismatchError(
                f"Expected shape {self._shape}, "
                f"got {tuple(original.shape)}."
            )
        reconstructed = self.reconstruct()
        original_device = original.to(
            device=self._device, dtype=self._dtype
        )
        error = torch.norm(original_device - reconstructed).item()
        norm = torch.norm(original_device).item()
        if norm < 1e-12:
            return 0.0
        return float(error / norm)


class CPDecomposition(TensorDecompositionBase):
    """Canonical Polyadic (CP / CANDECOMP/PARAFAC) decomposition.

    Decomposes a tensor into a sum of rank-1 components:

        X ≈ Σᵣ λᵣ · a_r ⊗ b_r ⊗ c_r ...

    Uses Alternating Least Squares (ALS) for fitting.

    Args:
        shape: Shape of the input tensor.
        rank: Number of rank-1 components.
        max_iter: Maximum ALS iterations.
        tol: Convergence tolerance on relative change.
        device: Torch device.
        dtype: Data type.

    Example:
        >>> cp = CPDecomposition(shape=(10, 10, 10), rank=3)
        >>> data = torch.randn(10, 10, 10)
        >>> cp.decompose(data)
        >>> approx = cp.reconstruct()
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        rank: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(shape, rank)
        self._max_iter = max_iter
        self._tol = tol
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._factors: List[Tensor] = []
        self._weights: Optional[Tensor] = None
        self._decomposed = False

    @property
    def factors(self) -> List[Tensor]:
        """Factor matrices from CP decomposition, shape ``(I_n, R)``."""
        if not self._decomposed:
            raise DecompositionError(
                "No decomposition performed yet. Call decompose() first."
            )
        return self._factors

    @property
    def weights(self) -> Tensor:
        """Component weights (lambdas) of shape ``(R,)``."""
        if self._weights is None:
            raise DecompositionError(
                "No decomposition performed yet. Call decompose() first."
            )
        return self._weights

    def decompose(self, tensor: Tensor) -> None:
        """Fit CP decomposition via Alternating Least Squares.

        Args:
            tensor: Input tensor of shape ``self.shape``.

        Raises:
            DimensionsMismatchError: If shapes do not match.
        """
        if tuple(tensor.shape) != self._shape:
            raise DimensionsMismatchError(
                f"Expected shape {self._shape}, "
                f"got {tuple(tensor.shape)}."
            )
        tensor = tensor.to(device=self._device, dtype=self._dtype)
        n_modes = tensor.dim()
        rank = self._rank

        # Initialize factors randomly
        self._factors = [
            torch.randn(
                self._shape[m],
                rank,
                device=self._device,
                dtype=self._dtype,
            )
            for m in range(n_modes)
        ]

        prev_error = float("inf")

        for _iteration in range(self._max_iter):
            for mode in range(n_modes):
                # Khatri-Rao product of all factors except current mode
                kr = self._khatri_rao_except(mode)
                # Unfold tensor along current mode
                unfolded = PolyadicTensor._mode_n_unfold(tensor, mode)
                # Solve least-squares: unfolded ≈ factor[mode] @ kr.T
                gram = kr.T @ kr  # (R, R)
                rhs = unfolded @ kr  # (I_mode, R)
                try:
                    self._factors[mode] = torch.linalg.solve(
                        gram.T, rhs.T
                    ).T
                except torch.linalg.LinAlgError:  # type: ignore[attr-defined]
                    # Fallback to pseudoinverse
                    self._factors[mode] = rhs @ torch.linalg.pinv(gram)

            # Normalize factors and extract weights
            self._weights = torch.ones(
                rank, device=self._device, dtype=self._dtype
            )
            for m in range(n_modes):
                norms = torch.norm(self._factors[m], dim=0)
                norms = torch.clamp(norms, min=1e-12)
                self._factors[m] = self._factors[m] / norms
                self._weights = self._weights * norms

            # Check convergence
            recon = self.reconstruct()
            error = torch.norm(tensor - recon).item()
            rel_change = abs(prev_error - error) / max(
                abs(prev_error), 1e-12
            )
            if rel_change < self._tol:
                break
            prev_error = error

        self._decomposed = True

    def _khatri_rao_except(self, skip_mode: int) -> Tensor:
        """Compute Khatri-Rao product of all factors except one.

        Args:
            skip_mode: Mode index to exclude.

        Returns:
            Khatri-Rao product matrix of shape
            ``(prod(I_n for n != skip_mode), R)``.
        """
        modes = [
            m for m in range(len(self._factors)) if m != skip_mode
        ]
        result = self._factors[modes[0]]
        for m in modes[1:]:
            result = self._khatri_rao(result, self._factors[m])
        return result

    @staticmethod
    def _khatri_rao(a: Tensor, b: Tensor) -> Tensor:
        """Column-wise Kronecker (Khatri-Rao) product.

        Args:
            a: Matrix of shape ``(I, R)``.
            b: Matrix of shape ``(J, R)``.

        Returns:
            Matrix of shape ``(I*J, R)``.
        """
        i_rows, rank = a.shape
        j_rows = b.shape[0]
        result = a.unsqueeze(1) * b.unsqueeze(0)  # (I, J, R)
        return result.reshape(i_rows * j_rows, rank)

    def reconstruct(self) -> Tensor:
        """Reconstruct tensor from CP factors and weights.

        Returns:
            Tensor approximation of shape ``self.shape``.

        Raises:
            DecompositionError: If decomposition has not been performed.
        """
        if not self._factors:
            raise DecompositionError(
                "No decomposition performed yet. Call decompose() first."
            )

        result = torch.zeros(
            self._shape, device=self._device, dtype=self._dtype
        )
        weights = (
            self._weights
            if self._weights is not None
            else torch.ones(
                self._rank, device=self._device, dtype=self._dtype
            )
        )

        for r in range(min(self._rank, self._factors[0].shape[1])):
            component = weights[r]
            outer = self._factors[0][:, r]
            for m in range(1, len(self._factors)):
                outer = torch.outer(outer.flatten(), self._factors[m][:, r])
                outer = outer.flatten()
            result += component * outer.reshape(self._shape)
        return result
