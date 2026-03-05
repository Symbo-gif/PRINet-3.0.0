"""Adversarial attack utilities for PRINet tracker evaluation.

Implements FGSM and PGD attacks for evaluating robustness of
PhaseTracker vs SlotAttention-based trackers under adversarial
perturbations to input detection features.

References:
    - Goodfellow et al. (2015), "Explaining and Harnessing Adversarial Examples"
    - Madry et al. (2018), "Towards Deep Learning Models Resistant to Adversarial Attacks"
    - AKOrN (ICLR 2025), Kuramoto oscillators achieve 58.9% adversarial accuracy
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor


def fgsm_attack(
    model: nn.Module,
    dets_t: Tensor,
    dets_t1: Tensor,
    n_objects: int,
    epsilon: float,
    loss_fn: Optional[object] = None,
) -> Tensor:
    """Fast Gradient Sign Method (FGSM) attack on input detections.

    Perturbs ``dets_t`` to maximize tracking loss using single-step
    gradient sign perturbation.

    Args:
        model: Tracker model with ``forward(dets_t, dets_t1)`` interface.
        dets_t: Frame t detections ``(N, D)``.
        dets_t1: Frame t+1 detections ``(N, D)``.
        n_objects: Number of ground-truth objects.
        epsilon: Perturbation magnitude (L-inf bound).
        loss_fn: Optional custom loss function. If None, uses
            cross-entropy on similarity matrix.

    Returns:
        Perturbed detections ``(N, D)`` of same shape as ``dets_t``.
    """
    model.eval()
    dets_t_adv = dets_t.clone().detach().requires_grad_(True)

    _, sim = model(dets_t_adv, dets_t1)
    N = min(sim.shape[0], sim.shape[1], n_objects)
    if N == 0:
        return dets_t.clone()

    sim_block = sim[:N, :N]
    target = torch.arange(N, device=sim.device)
    # Maximize loss = minimize negative loss
    loss = torch.nn.functional.cross_entropy(sim_block / 0.1, target)
    loss.backward()  # type: ignore[no-untyped-call]

    if dets_t_adv.grad is None:
        return dets_t.clone()

    perturbation = epsilon * dets_t_adv.grad.sign()
    perturbed = (dets_t + perturbation).detach()
    return perturbed


def pgd_attack(
    model: nn.Module,
    dets_t: Tensor,
    dets_t1: Tensor,
    n_objects: int,
    epsilon: float,
    alpha: Optional[float] = None,
    steps: int = 20,
    random_start: bool = True,
    seed: int = 42,
) -> Tensor:
    """Projected Gradient Descent (PGD) attack on input detections.

    Iteratively perturbs ``dets_t`` to maximize tracking loss,
    projecting back to the epsilon-ball after each step.

    Args:
        model: Tracker model with ``forward(dets_t, dets_t1)`` interface.
        dets_t: Frame t detections ``(N, D)``.
        dets_t1: Frame t+1 detections ``(N, D)``.
        n_objects: Number of ground-truth objects.
        epsilon: Perturbation magnitude (L-inf bound).
        alpha: Step size per iteration. Defaults to ``epsilon / 4``.
        steps: Number of PGD steps.
        random_start: If True, initialize with random perturbation.
        seed: Random seed for random start.

    Returns:
        Perturbed detections ``(N, D)`` of same shape as ``dets_t``.
    """
    if alpha is None:
        alpha = epsilon / 4.0

    model.eval()
    dets_orig = dets_t.clone().detach()

    # Random initialization within epsilon-ball
    if random_start:
        gen = torch.Generator(device=dets_t.device)
        gen.manual_seed(seed)
        delta = torch.empty_like(dets_t).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(dets_t)

    for _ in range(steps):
        dets_adv = (dets_orig + delta).requires_grad_(True)
        _, sim = model(dets_adv, dets_t1)
        N = min(sim.shape[0], sim.shape[1], n_objects)
        if N == 0:
            break

        sim_block = sim[:N, :N]
        target = torch.arange(N, device=sim.device)
        loss = torch.nn.functional.cross_entropy(sim_block / 0.1, target)
        loss.backward()  # type: ignore[no-untyped-call]

        if dets_adv.grad is None:
            break

        # Gradient ascent step
        delta = delta + alpha * dets_adv.grad.sign()
        # Project back to L-inf epsilon-ball
        delta = delta.clamp(-epsilon, epsilon)
        delta = delta.detach()

    return (dets_orig + delta).detach()


def adversarial_evaluate(
    model: nn.Module,
    dataset: list[Any],
    epsilon: float,
    attack_fn: str = "fgsm",
    pgd_steps: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate model IP under adversarial attack.

    Args:
        model: Tracker model.
        dataset: List of SequenceData objects.
        epsilon: Attack strength.
        attack_fn: ``"fgsm"`` or ``"pgd"``.
        pgd_steps: Number of PGD steps (if PGD).
        seed: Random seed.

    Returns:
        Dict with ``clean_ip``, ``adv_ip``, ``degradation``,
        ``per_seq_clean``, ``per_seq_adv``.
    """
    model.eval()
    clean_ips = []
    adv_ips = []

    for i, seq in enumerate(dataset):
        frames = [f.to(next(model.parameters()).device) for f in seq.frames]

        # Clean evaluation
        with torch.no_grad():
            result = model.track_sequence(frames)  # type: ignore[operator]
            clean_ips.append(result["identity_preservation"])

        # Adversarial evaluation: perturb each frame
        adv_frames = []
        for t in range(len(frames)):
            if t == 0:
                adv_frames.append(frames[t])
                continue
            if attack_fn == "fgsm":
                adv_f = fgsm_attack(
                    model,
                    frames[t - 1],
                    frames[t],
                    seq.n_objects,
                    epsilon,
                )
            else:
                adv_f = pgd_attack(
                    model,
                    frames[t - 1],
                    frames[t],
                    seq.n_objects,
                    epsilon,
                    steps=pgd_steps,
                    seed=seed + i * 100 + t,
                )
            adv_frames.append(adv_f)

        with torch.no_grad():
            adv_result = model.track_sequence(adv_frames)  # type: ignore[operator]
            adv_ips.append(adv_result["identity_preservation"])

    import numpy as np

    clean_arr = np.array(clean_ips)
    adv_arr = np.array(adv_ips)

    return {
        "clean_ip": float(clean_arr.mean()),
        "adv_ip": float(adv_arr.mean()),
        "degradation": float(clean_arr.mean() - adv_arr.mean()),
        "per_seq_clean": clean_ips,
        "per_seq_adv": adv_ips,
    }


def adversarial_comparison(
    pt_model: nn.Module,
    sa_model: nn.Module,
    dataset: list[Any],
    epsilons: list[float],
    attack_types: list[str],
    pgd_steps: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Side-by-side adversarial robustness comparison.

    Args:
        pt_model: PhaseTracker model.
        sa_model: SlotAttention model.
        dataset: List of SequenceData objects.
        epsilons: List of epsilon values to test.
        attack_types: List of attack types (``"fgsm"``, ``"pgd"``).
        pgd_steps: PGD steps.
        seed: Random seed.

    Returns:
        Dict with per-epsilon, per-attack results for both models.
    """
    results: dict[str, Any] = {}
    for attack in attack_types:
        results[attack] = {}
        for eps in epsilons:
            pt_res = adversarial_evaluate(
                pt_model,
                dataset,
                eps,
                attack,
                pgd_steps,
                seed,
            )
            sa_res = adversarial_evaluate(
                sa_model,
                dataset,
                eps,
                attack,
                pgd_steps,
                seed,
            )
            results[attack][str(eps)] = {
                "epsilon": eps,
                "pt_clean_ip": pt_res["clean_ip"],
                "pt_adv_ip": pt_res["adv_ip"],
                "pt_degradation": pt_res["degradation"],
                "sa_clean_ip": sa_res["clean_ip"],
                "sa_adv_ip": sa_res["adv_ip"],
                "sa_degradation": sa_res["degradation"],
            }
    return results
