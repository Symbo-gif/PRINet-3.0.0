"""GPU Benchmarks — Run all PRINet benchmarks on CUDA.

Runs oscillator scaling, desync catastrophe, and MNIST subset benchmarks
on GPU (CUDA) and saves combined results to
Docs/test_and_benchmark_results/benchmark_gpu_results.json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prinet.core.measurement import kuramoto_order_parameter
from prinet.core.propagation import KuramotoOscillator, OscillatorState
from prinet.nn.layers import PRINetModel
from prinet.nn.optimizers import SynchronizedGradientDescent

SEED = 42
DEVICE = torch.device("cuda")


# ===================================================================
# 1. Oscillator Scaling on GPU
# ===================================================================

def benchmark_oscillator_scaling_gpu() -> list[dict]:
    """Run oscillator simulation at multiple scales on GPU."""
    sizes = [100, 500, 1_000, 5_000, 10_000]
    results: list[dict] = []

    for n in sizes:
        torch.manual_seed(SEED)
        use_mean_field = n >= 1_000

        model = KuramotoOscillator(
            n_oscillators=n,
            coupling_strength=2.0 / n,
            decay_rate=0.1,
            mean_field=use_mean_field,
            device=DEVICE,
        )

        state = OscillatorState.create_random(n, device=DEVICE, seed=SEED)

        n_steps = 1_000
        dt = 0.01

        # Warm-up for GPU
        if n == sizes[0]:
            _ = model.step(state, dt=dt)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        final_state, trajectory = model.integrate(
            state, n_steps=n_steps, dt=dt, method="rk4"
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        r = kuramoto_order_parameter(final_state.phase).item()
        entry = {
            "n_oscillators": n,
            "n_steps": n_steps,
            "dt": dt,
            "device": "cuda",
            "sparsity": "mean_field" if use_mean_field else "dense",
            "wall_seconds": round(elapsed, 4),
            "final_order_param": round(r, 6),
            "passed": elapsed < 10.0 if n == 10_000 else True,
        }
        results.append(entry)
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"  N={n:>6d}  t={elapsed:>8.4f}s  r={r:.4f}  [{status}]")

    return results


# ===================================================================
# 2. Desync Catastrophe on GPU
# ===================================================================

@dataclass
class DesyncResult:
    label: str
    order_params: list[float]
    final_order: float
    desynchronised: bool
    wall_seconds: float


def run_desync_without_barrier_gpu(
    n_oscillators: int = 64,
    n_steps: int = 200,
    lr: float = 0.1,
) -> DesyncResult:
    torch.manual_seed(SEED)
    model = PRINetModel(
        n_dims=16, n_resonances=n_oscillators, n_layers=2, n_concepts=10,
    ).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    kuramoto = KuramotoOscillator(
        n_oscillators=n_oscillators, coupling_strength=0.5, device=DEVICE,
    )
    state = OscillatorState.create_random(n_oscillators, device=DEVICE, seed=SEED)

    order_history: list[float] = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x = torch.randn(8, 16, device=DEVICE)
        target = torch.randint(0, 10, (8,), device=DEVICE)
        logits = model(x)
        loss = F.nll_loss(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        state = kuramoto.step(state, dt=0.01)
        r = kuramoto_order_parameter(state.phase).item()
        order_history.append(r)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    final = order_history[-1]
    return DesyncResult("plain_sgd_gpu", order_history, final, final < 0.5, elapsed)


def run_sync_with_barrier_gpu(
    n_oscillators: int = 64,
    n_steps: int = 200,
    lr: float = 0.1,
    lambda_sync: float = 0.5,
    k_critical: float = 0.6,
) -> DesyncResult:
    torch.manual_seed(SEED)
    model = PRINetModel(
        n_dims=16, n_resonances=n_oscillators, n_layers=2, n_concepts=10,
    ).to(DEVICE)
    kuramoto = KuramotoOscillator(
        n_oscillators=n_oscillators, coupling_strength=2.0, device=DEVICE,
    )
    opt = SynchronizedGradientDescent(
        model.parameters(), lr=lr, sync_penalty=lambda_sync, critical_order=k_critical,
    )
    state = OscillatorState.create_random(n_oscillators, device=DEVICE, seed=SEED)

    order_history: list[float] = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x = torch.randn(8, 16, device=DEVICE)
        target = torch.randint(0, 10, (8,), device=DEVICE)
        logits = model(x)
        loss = F.nll_loss(logits, target)
        opt.zero_grad()
        loss.backward()
        state = kuramoto.step(state, dt=0.01)
        r = kuramoto_order_parameter(state.phase).item()
        opt.step(order_parameter=r)
        order_history.append(r)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    final = order_history[-1]
    return DesyncResult("sync_sgd_barrier_gpu", order_history, final, final < 0.5, elapsed)


# ===================================================================
# 3. MNIST Subset on GPU
# ===================================================================

def benchmark_mnist_gpu() -> dict:
    N_OSCILLATORS = 32
    N_EPOCHS = 20
    BATCH_SIZE = 64
    SUBSET_SIZE = 1_000

    try:
        from torchvision import datasets, transforms
        tf = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds = datasets.MNIST(root="/tmp/mnist", train=True, download=True, transform=tf)
        torch.manual_seed(SEED)
        indices = torch.randperm(len(ds))[:SUBSET_SIZE]
        images = torch.stack([ds[i][0].flatten() for i in indices]).to(DEVICE)
        labels = torch.tensor([ds[i][1] for i in indices]).to(DEVICE)
        data_source = "real_mnist"
        print("  Using real MNIST data on GPU.")
    except ImportError:
        print("  torchvision not found — using synthetic data on GPU.")
        torch.manual_seed(SEED)
        images = torch.randn(SUBSET_SIZE, 784, device=DEVICE)
        labels = torch.randint(0, 10, (SUBSET_SIZE,), device=DEVICE)
        data_source = "synthetic"

    input_dim = images.shape[1]
    torch.manual_seed(SEED)
    model = PRINetModel(
        n_dims=input_dim, n_resonances=N_OSCILLATORS, n_layers=2, n_concepts=10,
    ).to(DEVICE)
    kuramoto = KuramotoOscillator(
        n_oscillators=N_OSCILLATORS, coupling_strength=2.0, device=DEVICE,
    )
    opt = SynchronizedGradientDescent(
        model.parameters(), lr=0.05, sync_penalty=0.3, critical_order=0.6,
    )
    state = OscillatorState.create_random(N_OSCILLATORS, device=DEVICE, seed=SEED)
    n_batches = max(1, SUBSET_SIZE // BATCH_SIZE)
    epoch_records: list[dict] = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for epoch in range(1, N_EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        for bi in range(n_batches):
            start = bi * BATCH_SIZE
            end = start + BATCH_SIZE
            xb, yb = images[start:end], labels[start:end]
            logits = model(xb)
            loss = F.nll_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            state = kuramoto.step(state, dt=0.01)
            r_step = kuramoto_order_parameter(state.phase).item()
            opt.step(order_parameter=r_step)
            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()

        r = kuramoto_order_parameter(state.phase).item()
        avg_loss = total_loss / n_batches
        accuracy = correct / min(SUBSET_SIZE, n_batches * BATCH_SIZE)
        rec = {"epoch": epoch, "loss": round(avg_loss, 4), "accuracy": round(accuracy, 4), "order_param": round(r, 4)}
        epoch_records.append(rec)
        print(f"  Epoch {epoch:>2d}/{N_EPOCHS}  loss={avg_loss:.4f}  acc={accuracy:.4f}  r={r:.4f}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    final_r = epoch_records[-1]["order_param"]
    passed = final_r > 0.8

    return {
        "benchmark": "mnist_subset_gpu",
        "data_source": data_source,
        "device": "cuda",
        "n_epochs": N_EPOCHS,
        "subset_size": SUBSET_SIZE,
        "final_order_param": final_r,
        "final_accuracy": epoch_records[-1]["accuracy"],
        "passed": passed,
        "wall_seconds": round(elapsed, 3),
        "epoch_records": epoch_records,
    }


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available!"
    gpu_name = torch.cuda.get_device_name(0)
    cuda_ver = torch.version.cuda
    torch_ver = torch.__version__

    print("=" * 70)
    print(f"  PRINet GPU BENCHMARKS")
    print(f"  GPU: {gpu_name}  |  CUDA: {cuda_ver}  |  PyTorch: {torch_ver}")
    print("=" * 70)

    # 1. Oscillator Scaling
    print("\n--- Oscillator Scaling (Task 1.7) ---")
    scaling_results = benchmark_oscillator_scaling_gpu()
    target_10k = next(r for r in scaling_results if r["n_oscillators"] == 10_000)
    print(f"  => 10k result: {target_10k['wall_seconds']:.4f}s  "
          f"[{'PASS' if target_10k['passed'] else 'FAIL'}]")

    # 2. Desync Catastrophe
    print("\n--- Desynchronization Catastrophe (Task 2.1) ---")
    desync_plain = run_desync_without_barrier_gpu()
    desync_sync = run_sync_with_barrier_gpu()
    print(f"  plain_sgd:        r={desync_plain.final_order:.4f}  "
          f"desync={'YES' if desync_plain.desynchronised else 'NO'}  "
          f"t={desync_plain.wall_seconds:.3f}s")
    print(f"  sync_sgd_barrier: r={desync_sync.final_order:.4f}  "
          f"desync={'YES' if desync_sync.desynchronised else 'NO'}  "
          f"t={desync_sync.wall_seconds:.3f}s")

    # 3. MNIST Subset
    print("\n--- MNIST Subset (Task 2.5) ---")
    mnist_result = benchmark_mnist_gpu()
    print(f"  => final r={mnist_result['final_order_param']:.4f}  "
          f"acc={mnist_result['final_accuracy']:.4f}  "
          f"[{'PASS' if mnist_result['passed'] else 'NOTE'}]  "
          f"t={mnist_result['wall_seconds']:.2f}s")

    # Compute VRAM usage
    vram_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    vram_reserved = torch.cuda.max_memory_reserved() / (1024**2)

    # Save combined results
    out_dir = Path(__file__).resolve().parents[1] / "Docs" / "test_and_benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "gpu_benchmarks": True,
        "environment": {
            "gpu": gpu_name,
            "cuda_version": cuda_ver,
            "pytorch_version": torch_ver,
            "python_version": sys.version.split()[0],
        },
        "memory": {
            "peak_allocated_mb": round(vram_allocated, 1),
            "peak_reserved_mb": round(vram_reserved, 1),
        },
        "oscillator_scaling": {
            "task": "1.7",
            "target": "10000 oscillators, 1000 steps, < 10 seconds",
            "results": scaling_results,
        },
        "desync_catastrophe": {
            "task": "2.1",
            "results": [
                {
                    "label": r.label,
                    "final_order": round(r.final_order, 6),
                    "desynchronised": r.desynchronised,
                    "wall_seconds": round(r.wall_seconds, 4),
                    "order_trace_first10": [round(x, 4) for x in r.order_params[:10]],
                    "order_trace_last10": [round(x, 4) for x in r.order_params[-10:]],
                }
                for r in (desync_plain, desync_sync)
            ],
        },
        "mnist_subset": mnist_result,
    }

    out_file = out_dir / "benchmark_gpu_results.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  VRAM Peak: {vram_allocated:.0f} MB allocated / {vram_reserved:.0f} MB reserved")
    print(f"  Results saved to {out_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
