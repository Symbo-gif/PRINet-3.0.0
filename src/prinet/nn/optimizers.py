"""Synchronized Optimizers for PRINet.

Implements optimizers that maintain phase synchronization during training
by incorporating barrier functions and synchronization penalties into
the gradient update. Core deliverable of Task 2.3 (Project 1B).

Example:
    >>> import torch
    >>> from prinet.nn.layers import PRINetModel
    >>> model = PRINetModel(n_resonances=32, n_dims=64, n_concepts=10)
    >>> optimizer = SynchronizedGradientDescent(
    ...     model.parameters(),
    ...     lr=0.01,
    ...     sync_penalty=0.1,
    ...     critical_order=0.5,
    ... )
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class SynchronizedGradientDescent(Optimizer):
    """SGD with synchronization barrier penalty (Project 1B).

    Extends standard SGD with a penalty term that prevents the Kuramoto
    order parameter from dropping below a critical threshold during
    training. The total loss becomes:

        L_total = L_task + λ · max(0, K_c - K)²

    where K is the current order parameter and K_c is the critical
    threshold.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        momentum: Momentum factor (default: 0).
        weight_decay: L2 regularization (default: 0).
        sync_penalty: Synchronization penalty weight λ.
        critical_order: Critical order parameter threshold K_c.
            Training is penalized when the order parameter drops
            below this value.
        dampening: Dampening for momentum (default: 0).

    Raises:
        ValueError: If any hyperparameter is out of valid range.

    Example:
        >>> params = [torch.randn(10, requires_grad=True)]
        >>> opt = SynchronizedGradientDescent(
        ...     params, lr=0.01, sync_penalty=0.1, critical_order=0.8
        ... )
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        sync_penalty: float = 0.1,
        critical_order: float = 0.5,
        dampening: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if sync_penalty < 0.0:
            raise ValueError(f"Invalid sync_penalty value: {sync_penalty}")
        if not 0.0 <= critical_order <= 1.0:
            raise ValueError(f"critical_order must be in [0, 1], got {critical_order}")

        defaults: Dict[str, float] = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            sync_penalty=sync_penalty,
            critical_order=critical_order,
            dampening=dampening,
        )
        super().__init__(params, defaults)

        # Track synchronization metrics across steps
        self._order_history: List[float] = []
        self._penalty_history: List[float] = []

    @property
    def order_history(self) -> List[float]:
        """History of order parameter values across optimization steps."""
        return self._order_history

    @property
    def penalty_history(self) -> List[float]:
        """History of synchronization penalty values."""
        return self._penalty_history

    def compute_sync_penalty(self, order_parameter: float) -> Tuple[float, float]:
        """Compute the synchronization barrier penalty.

        Args:
            order_parameter: Current Kuramoto order parameter r ∈ [0, 1].

        Returns:
            Tuple of (penalty_value, penalty_gradient_scale).
            The gradient scale is used to modulate learning rate.
        """
        k_c = self.defaults["critical_order"]
        lam = self.defaults["sync_penalty"]

        deficit = max(0.0, k_c - order_parameter)
        penalty = lam * deficit**2
        # Gradient of penalty w.r.t. order parameter
        grad_scale = 2.0 * lam * deficit if deficit > 0.0 else 0.0

        self._order_history.append(order_parameter)
        self._penalty_history.append(penalty)

        return penalty, grad_scale

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Optional[Callable[[], float]] = None,
        order_parameter: Optional[float] = None,
    ) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure: Callable returning the loss. Optional.
            order_parameter: Current Kuramoto order parameter.
                If provided, the synchronization penalty is applied.
                If ``None``, standard SGD is performed.

        Returns:
            Loss value if ``closure`` is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute sync penalty modulation
        penalty_value = 0.0
        grad_modulation = 1.0

        if order_parameter is not None:
            penalty_value, grad_scale = self.compute_sync_penalty(order_parameter)
            # Reduce learning rate when desynchronized (protective)
            if grad_scale > 0.0:
                grad_modulation = max(0.1, 1.0 - grad_scale)

        for group in self.param_groups:
            lr = group["lr"] * grad_modulation
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = buf

                p.add_(d_p, alpha=-lr)

        if loss is not None:
            return loss + penalty_value
        return loss


class RIPOptimizer(Optimizer):
    """Resonance-Induced Plasticity (RIP) optimizer.

    Implements the Hebbian-style update rule where coupling weights
    are updated based on phase coherence between connected oscillators:

        ΔKᵢⱼ = η · cos(φᵢ - φⱼ) · |rⱼ| · (r_target - rᵢ)

    This optimizer updates only coupling-type parameters.

    Args:
        params: Iterable of coupling parameters to optimize.
        lr: Learning rate η.
        target_amplitude: Target amplitude r_target for the RIP rule.

    Example:
        >>> coupling_params = [model.coupling]
        >>> rip = RIPOptimizer(coupling_params, lr=0.05)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        target_amplitude: float = 1.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if target_amplitude <= 0.0:
            raise ValueError(
                f"target_amplitude must be positive, " f"got {target_amplitude}"
            )
        defaults: Dict[str, float] = dict(lr=lr, target_amplitude=target_amplitude)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Optional[Callable[[], float]] = None,
        phase: Optional[Tensor] = None,
        amplitude: Optional[Tensor] = None,
    ) -> Optional[float]:
        """Perform a RIP update step.

        Args:
            closure: Optional loss closure.
            phase: Current phase tensor ``(N,)`` or ``(B, N)`` for
                computing Hebbian updates.
            amplitude: Current amplitude tensor, same shape as phase.

        Returns:
            Loss value if ``closure`` is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            r_target = group["target_amplitude"]

            for p in group["params"]:
                if p.grad is not None:
                    # Standard gradient descent component
                    p.add_(p.grad, alpha=-lr)

                # RIP Hebbian update if phase/amplitude provided
                if (
                    phase is not None
                    and amplitude is not None
                    and p.dim() == 2
                    and p.shape[0] == p.shape[1]
                ):
                    n = p.shape[0]
                    if phase.shape[-1] == n:
                        ph = phase.mean(dim=0) if phase.dim() > 1 else phase
                        amp = (
                            amplitude.mean(dim=0) if amplitude.dim() > 1 else amplitude
                        )

                        # cos(φᵢ - φⱼ)
                        phase_diff = ph.unsqueeze(0) - ph.unsqueeze(1)
                        cos_coherence = torch.cos(phase_diff)

                        # |rⱼ| broadcast
                        amp_j = amp.unsqueeze(0)

                        # (r_target - rᵢ) broadcast
                        deficit = r_target - amp.unsqueeze(1)

                        # RIP update
                        delta = lr * cos_coherence * amp_j * deficit
                        p.add_(delta)

                        # Zero diagonal (no self-coupling)
                        p.fill_diagonal_(0.0)

        return loss


class SCALROptimizer(Optimizer):
    """Synchronization-Coupled Adaptive Learning Rate (SCALR) optimizer.

    Scales the learning rate based on the Kuramoto order parameter:

        η_eff(t) = η_base · f(r(t))

    where ``r(t)`` is the instantaneous Kuramoto order parameter and
    ``f`` is a scaling function. When oscillators are well-synchronized
    (high ``r``), the optimizer takes larger steps; when desynchronized
    (low ``r``), it slows down to let the dynamics stabilize.

    The scaling function is:

        f(r) = r_min + (1 - r_min) · r^α

    where ``r_min`` is the minimum learning rate fraction (to prevent
    stalling) and ``α`` controls sensitivity to synchronization.

    Q3 enhancements:
        - **Oscillation-aware decay**: Detects windowed variance in ``r(t)``
          and applies exponential lr decay when destabilizing oscillations
          are detected.
        - **Per-frequency lr scaling**: Accepts per-parameter-group order
          parameters for hierarchical models.
        - **Adaptive r_min**: Automatically adjusts ``r_min`` via EMA of
          the order parameter.

    Args:
        params: Iterable of parameters to optimize.
        lr: Base learning rate η_base.
        momentum: Momentum factor.
        weight_decay: L2 regularization.
        r_min: Minimum learning rate fraction when r → 0.
        alpha: Exponent controlling synchronization sensitivity.
            α = 1 gives linear scaling; α > 1 is more aggressive.
        warmup_steps: Number of initial steps to use full base LR
            before SCALR kicks in (avoids cold-start issues).
        oscillation_window: Window size for oscillation detection.
        oscillation_threshold: Variance threshold for oscillation detection.
        oscillation_decay: Multiplicative lr decay factor when
            oscillations are detected. Applied once per detection.
        adaptive_r_min: If ``True``, auto-adjust ``r_min`` based on
            EMA of the order parameter.
        r_min_ema_alpha: EMA smoothing factor for adaptive r_min.

    Example:
        >>> model = PRINetModel()
        >>> opt = SCALROptimizer(model.parameters(), lr=0.01, alpha=1.5)
        >>> # In training loop:
        >>> opt.step(order_parameter=r_value)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        r_min: float = 0.1,
        alpha: float = 1.0,
        warmup_steps: int = 0,
        oscillation_window: int = 20,
        oscillation_threshold: float = 0.01,
        oscillation_decay: float = 0.95,
        adaptive_r_min: bool = False,
        r_min_ema_alpha: float = 0.1,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= r_min <= 1.0:
            raise ValueError(f"r_min must be in [0, 1], got {r_min}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

        defaults: Dict[str, float] = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self._r_min = r_min
        self._alpha = alpha
        self._warmup_steps = warmup_steps
        self._step_count = 0

        # Q3: Oscillation-aware decay
        self._oscillation_window = oscillation_window
        self._oscillation_threshold = oscillation_threshold
        self._oscillation_decay = oscillation_decay
        self._lr_decay_factor = 1.0  # cumulative decay

        # Q3: Adaptive r_min
        self._adaptive_r_min = adaptive_r_min
        self._r_min_ema_alpha = r_min_ema_alpha
        self._r_ema: Optional[float] = None

        # Tracking history
        self._lr_history: List[float] = []
        self._order_history: List[float] = []

    @property
    def lr_history(self) -> List[float]:
        """History of effective learning rates."""
        return self._lr_history

    @property
    def order_history(self) -> List[float]:
        """History of order parameter values."""
        return self._order_history

    def compute_lr_scale(self, order_parameter: float) -> float:
        """Compute the learning rate scaling factor.

        Args:
            order_parameter: Current Kuramoto order parameter r ∈ [0, 1].

        Returns:
            Scaling factor in [r_min, 1.0].
        """
        r = max(0.0, min(1.0, order_parameter))
        return float(self._r_min + (1.0 - self._r_min) * (r**self._alpha))

    def _detect_oscillation(self) -> bool:
        """Check if recent order-parameter history shows oscillation.

        Returns:
            ``True`` if windowed variance exceeds threshold.
        """
        w = self._oscillation_window
        if len(self._order_history) < w:
            return False
        recent = self._order_history[-w:]
        mean = sum(recent) / len(recent)
        var = sum((v - mean) ** 2 for v in recent) / len(recent)
        return var > self._oscillation_threshold

    def _update_adaptive_r_min(self, order_parameter: float) -> None:
        """Update adaptive r_min via EMA of order parameter.

        Args:
            order_parameter: Current order parameter.
        """
        if self._r_ema is None:
            self._r_ema = order_parameter
        else:
            a = self._r_min_ema_alpha
            self._r_ema = a * order_parameter + (1.0 - a) * self._r_ema
        # Set r_min to a fraction of the EMA (floor at 0.01)
        self._r_min = max(0.01, min(0.5, self._r_ema * 0.3))

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Optional[Callable[[], float]] = None,
        order_parameter: Optional[Union[float, Dict[str, float]]] = None,
    ) -> Optional[float]:
        """Perform a SCALR optimization step.

        Args:
            closure: Optional loss closure.
            order_parameter: Current Kuramoto order parameter r ∈ [0, 1].
                Can be a single float (global) or a dict mapping
                param-group names to per-group order parameters for
                per-frequency lr scaling. If ``None``, uses full base
                learning rate.

        Returns:
            Loss value if ``closure`` is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        # Resolve per-group order parameters
        if isinstance(order_parameter, dict):
            global_r = sum(order_parameter.values()) / max(len(order_parameter), 1)
        elif order_parameter is not None:
            global_r = order_parameter
        else:
            global_r = None

        # Record history
        if global_r is not None:
            self._order_history.append(global_r)

        # Q3: Adaptive r_min
        if self._adaptive_r_min and global_r is not None:
            self._update_adaptive_r_min(global_r)

        # Q3: Oscillation-aware decay
        if self._detect_oscillation():
            self._lr_decay_factor *= self._oscillation_decay

        for group_idx, group in enumerate(self.param_groups):
            # Determine order parameter for this group
            if isinstance(order_parameter, dict):
                group_name = group.get("name", str(group_idx))
                group_r = order_parameter.get(group_name, global_r)
            else:
                group_r = global_r if order_parameter is not None else None

            # Compute effective LR scale
            if group_r is not None and self._step_count > self._warmup_steps:
                lr_scale = self.compute_lr_scale(group_r)
            else:
                lr_scale = 1.0

            # Apply cumulative oscillation decay
            lr = group["lr"] * lr_scale * self._lr_decay_factor
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            self._lr_history.append(lr)

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1.0)
                    d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss
