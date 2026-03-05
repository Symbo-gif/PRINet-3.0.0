# Contributing to PRINet

Thank you for your interest in contributing to PRINet! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## How to Contribute

### Reporting Bugs

Before filing a bug report, please:
1. Check the [existing issues](https://github.com/michaelmaillet/prinet/issues) to avoid duplicates.
2. Use the latest version of PRINet.
3. Verify you can reproduce the issue with the minimum code possible.

When filing a bug report, include:
- Python version (`python --version`)
- PRINet version (`pip show prinet`)
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- CUDA version (if applicable)
- Complete error traceback
- Minimal reproducible example

### Feature Requests

Open an issue with the `enhancement` label. Describe:
- The use case that motivates the feature
- Proposed API or interface
- Whether you'd like to implement it yourself

### Pull Requests

1. **Fork** the repository and create your branch from `main`:
   ```bash
   git checkout -b feat/my-new-feature
   ```

2. **Install** in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Write tests** for your changes. PRINet requires ≥ 95% coverage for new code.

4. **Run the test suite** and ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```

5. **Follow the coding standards**:
   - PEP 8 compliance (enforced by `black` and `isort`)
   - Type hints on all public functions
   - Google-style docstrings on all public symbols
   - No bare `except:` clauses

6. **Format your code**:
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

7. **Run type checking**:
   ```bash
   mypy src/prinet --strict
   ```

8. **Update documentation** if you added or changed any public API.

9. **Submit the PR** with:
   - A clear description of changes
   - References to related issues
   - Test results summary

---

## Development Setup

```bash
# Clone the repository
git clone https://github.com/michaelmaillet/prinet.git
cd prinet

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/macOS

# Install with all development extras
pip install -e ".[all]"

# Verify installation
pytest tests/ -v -m "not slow and not gpu"
```

### For GPU Development (Windows + CUDA)

```bash
# Install CUDA-enabled PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Then install PRINet and its extras
pip install -e ".[dev,mot]"

# Run GPU tests
pytest tests/test_gpu.py -v -m gpu
```

---

## Coding Standards

PRINet follows strict coding standards documented in the project:

### Style
- **Formatter**: `black` (line length 88)
- **Import ordering**: `isort`
- **Type checking**: `mypy --strict`
- **Docstrings**: Google Style

### Example Function

```python
def compute_binding_strength(
    phases: torch.Tensor,
    coupling_matrix: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the pairwise oscillator binding strength.

    Uses normalised phase coherence weighted by the coupling topology.

    Args:
        phases: Oscillator phase angles in radians. Shape: (N,).
        coupling_matrix: Symmetric adjacency matrix. Shape: (N, N).
        temperature: Softmax temperature for binding sharpness. Must be > 0.

    Returns:
        Pairwise binding strength matrix. Shape: (N, N), values in [0, 1].

    Raises:
        ValueError: If phases and coupling_matrix dimensions are inconsistent.
        ValueError: If temperature <= 0.

    Examples:
        >>> phases = torch.rand(8) * 2 * torch.pi
        >>> W = torch.ones(8, 8) / 8
        >>> strength = compute_binding_strength(phases, W)
        >>> strength.shape
        torch.Size([8, 8])
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    n = phases.shape[0]
    if coupling_matrix.shape != (n, n):
        raise ValueError(
            f"coupling_matrix shape {coupling_matrix.shape} inconsistent "
            f"with phases shape {phases.shape}"
        )
    phase_diff = phases.unsqueeze(1) - phases.unsqueeze(0)  # (N, N)
    coherence = torch.cos(phase_diff)  # (N, N)
    return torch.softmax(coherence * coupling_matrix / temperature, dim=-1)
```

---

## Commit Messages

PRINet uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add community topology coupling mode
fix: correct phase wrap-around in ring coupling
docs: add OscilloSim chimera tutorial
test: add GPU test for fused Triton kernel
perf: 2x speedup in local order parameter computation
refactor: extract binding logic into separate module
chore: bump torch requirement to >=2.1
```

---

## Areas Needing Contributions

| Area | Difficulty | Description |
|---|---|---|
| Coupling topologies | Medium | Add power-law or scale-free graph coupling |
| Multi-GPU support | Hard | Data-parallel OscilloSim across multiple GPUs |
| macOS / Linux CI | Medium | Add GitHub Actions runners for non-Windows |
| New benchmarks | Easy | Add real-world MOT dataset evaluation |
| Documentation | Easy | Improve docstring examples and tutorials |
| Triton kernels (Linux) | Medium | Extend Triton kernel coverage for Linux |

---

## Questions?

Open an issue with the `question` label or email **therealmichaelmaillet@gmail.com**.
