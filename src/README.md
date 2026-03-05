# src/

Source root for the PRINet Python package.

## Structure

```
src/
└── prinet/          # Main package (install from source via pip install -e .)
    ├── core/        # Oscillator dynamics, decomposition, measurement
    ├── nn/          # PyTorch neural network modules and optimizers
    └── utils/       # GPU kernels, simulators, datasets, utilities
```

## Build

The package is built via [setuptools](https://setuptools.pypa.io/) with configuration in [`pyproject.toml`](../pyproject.toml). The `src/` layout follows [PEP 517](https://peps.python.org/pep-0517/) best practices.

```bash
# Editable install for development
pip install -e ".[dev]"
```

## License

MIT — see [LICENSE](../LICENSE).
