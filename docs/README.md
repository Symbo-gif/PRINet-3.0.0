# docs/

Sphinx documentation source for PRINet, published at [prinet.readthedocs.io](https://prinet.readthedocs.io).

## Contents

### Sphinx Configuration

| File | Description |
|---|---|
| `conf.py` | Sphinx configuration — theme, extensions, autodoc settings |
| `index.rst` | Documentation root — table of contents and landing page |
| `requirements.txt` | Sphinx build dependencies (sphinx, myst-parser, rtd-theme) |

### User Guides

| File | Description |
|---|---|
| `Getting_Started_Tutorial.md` | Step-by-step tutorial for new users |
| `Architecture_Guide.md` | PRINet architecture overview and design rationale |
| `API_Reference_Coupling_Topologies.md` | Detailed reference for all 8 OscilloSim coupling modes |
| `Capacity_Analysis.md` | Binding capacity analysis and theoretical limits |
| `getting_started.rst` | RST version of getting started guide (Sphinx toctree) |
| `architecture.rst` | RST version of architecture guide (Sphinx toctree) |
| `changelog.rst` | RST changelog (links to CHANGELOG.md) |

### API Reference

| File | Description |
|---|---|
| `api/core.rst` | `prinet.core` API autodoc reference |
| `api/nn.rst` | `prinet.nn` API autodoc reference |
| `api/utils.rst` | `prinet.utils` API autodoc reference |

### Sphinx Assets

| Directory | Description |
|---|---|
| `_static/` | Static assets (CSS overrides, images) |
| `_templates/` | Jinja2 template overrides |

## Building Documentation Locally

```bash
# Install docs dependencies
pip install -r docs/requirements.txt

# Build HTML documentation
cd docs/
sphinx-build -b html . _build/html

# Open in browser
open _build/html/index.html   # macOS
start _build/html/index.html  # Windows
```

## ReadTheDocs

The documentation is automatically built and published via ReadTheDocs on every push to `main`. Configuration is in [`.readthedocs.yaml`](../.readthedocs.yaml).

## License

MIT — see [LICENSE](../LICENSE).
