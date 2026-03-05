# .github/

GitHub configuration files for PRINet.

## Workflows

CI/CD pipelines in [`.github/workflows/`](workflows/):

| Workflow | File | Triggers | Description |
|---|---|---|---|
| **PRINet CI** | `ci.yml` | Push/PR to `main`, `develop` | CPU tests (Python 3.11–3.13), Windows tests, type checking (mypy), GPU tests (self-hosted), reproducibility check |
| **Code Quality** | `lint.yml` | Push/PR to `main`, `develop` | Black formatting, isort import ordering, Bandit SAST, pip-audit dependency CVE scan |
| **Release to PyPI** | `release.yml` | Tag `v*.*.*` or manual dispatch | Build sdist+wheel, publish to TestPyPI then PyPI (Trusted Publishing / OIDC), create GitHub Release with changelog |

## CI Matrix

| Job | Runner | Python | Purpose |
|---|---|---|---|
| CPU Tests | `ubuntu-latest` | 3.11, 3.12, 3.13 | Core test suite, coverage upload |
| Windows Tests | `windows-latest` | 3.12 | Platform-specific validation |
| Type Check | `ubuntu-latest` | 3.12 | `mypy --strict` |
| GPU Tests | `self-hosted, gpu` | 3.12 | Full suite with CUDA (on main or `[gpu]` tag) |
| Reproducibility | `ubuntu-latest` | 3.12 | Verify `reproduce.py` generates expected outputs |
| Format Check | `ubuntu-latest` | 3.12 | `black --check`, `isort --check-only` |
| Security Audit | `ubuntu-latest` | 3.12 | `bandit`, `pip-audit` |

## License

MIT — see [LICENSE](../LICENSE).
