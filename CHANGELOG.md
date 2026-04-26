# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-04-26

### Added
- Initial public release of PRINet on PyPI.
- Core phase-resonance dynamics: CP decomposition, propagation, and measurement
  primitives in `prinet.core`.
- PyTorch-compatible layers, models, and optimizers in `prinet.nn`.
- GPU-accelerated solvers, profiling, and dataset utilities in `prinet.utils`.
- `py.typed` marker for downstream type checking.
- Reproducibility entry point (`reproduce.py`) that regenerates all paper figures
  from stored benchmark artefacts.

### Packaging
- Trusted Publishing (OIDC) workflow to TestPyPI and PyPI.
- Wheel and sdist build verified with `twine check`.
- Supports Python 3.11, 3.12, and 3.13 on Linux, macOS, and Windows.

[3.0.0]: https://github.com/Symbo-gif/PRINet-3.0.0/releases/tag/v3.0.0
