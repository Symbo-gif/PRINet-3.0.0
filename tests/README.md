# tests/

Test suite for the PRINet package — **1,626 tests, 0 failures**.

## Running Tests

```bash
# Full test suite (CPU, ~90 seconds)
pytest tests/ -v

# Skip slow training tests
pytest tests/ -v -m "not slow"

# GPU tests only (requires CUDA)
pytest tests/test_gpu.py -v -m gpu

# With coverage report
pytest tests/ --cov=prinet --cov-report=html
```

## Test Files

### Core Tests

| File | Description |
|---|---|
| `test_core.py` | Core oscillator dynamics, decomposition, measurement functions |
| `test_nn.py` | Neural network layers, architectures, model construction |
| `test_utils.py` | Utility functions, OscilloSim, dataset loaders |
| `test_gpu.py` | GPU/CUDA-specific tests (skipped on CPU) |

### Feature-Specific Tests

| File | Description |
|---|---|
| `test_clevr_n.py` | CLEVR-N dataset generation and binding evaluation |
| `test_hierarchical.py` | Hierarchical coupling topology tests |
| `test_hybrid.py` | HybridPRINetV2 architecture tests |
| `test_phases.py` | Phase dynamics and synchronization tests |
| `test_phase_to_rate.py` | Phase-to-rate conversion and autoencoder tests |
| `test_scalr_enhanced.py` | SCALR optimizer convergence and scheduling |
| `test_subconscious.py` | SubconsciousController and CognitiveDaemon tests |
| `test_triton_kernels.py` | Triton fused kernel correctness (skipped without Triton) |

### Integration Tests

| File | Description |
|---|---|
| `test_integration_q3.py` | End-to-end integration tests |
| `test_q2.py`, `test_q2_remaining.py` | Q2 feature integration |
| `test_q3_new.py` | Q3 new feature tests |

### Research Phase Tests

Tests organized by research development phase:

| Phase | Files |
|---|---|
| Year 2 | `test_y2q1.py`, `test_y2q2.py`, `test_y2q3.py`, `test_y2q4.py` |
| Year 3 | `test_y3q1.py`, `test_y3q2.py`, `test_y3q3.py`, `test_y3q4.py`, `test_y3q45.py`, `test_y3q49.py` |
| Year 4 | `test_y4q1.py` through `test_y4q1_9.py`, `test_y4q2.py`, `test_y4q3.py`, `test_y4q4.py` |

## Test Markers

| Marker | Description |
|---|---|
| `gpu` | Requires CUDA GPU — skipped on CPU-only machines |
| `slow` | Takes >5 seconds (training convergence, benchmarks) |

## Coverage

Target: ≥95% code coverage. Generate an HTML report with:

```bash
pytest tests/ --cov=prinet --cov-report=html
open htmlcov/index.html
```

## License

MIT — see [LICENSE](../LICENSE).
