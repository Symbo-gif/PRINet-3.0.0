PRINet Documentation
====================

**PRINet** (Polyadic Resonance Inference Network) is a novel neural network
architecture based on resonance dynamics, polyadic tensor decomposition, and
phase synchronization. It processes information via multi-dimensional harmonic
resonances rather than token embeddings.

.. code-block:: python

   import torch
   from prinet import PRINetModel

   model = PRINetModel(n_resonances=64, n_dims=256, n_concepts=10)
   x = torch.randn(32, 256)
   log_probs = model(x)   # [32, 10]

Key Features
------------

- **Oscillatory binding**: Objects are represented as synchronized phase
  clusters in a network of coupled oscillators.
- **Multi-rate integration**: Delta, theta, and gamma frequency bands operate
  at different timescales for hierarchical processing.
- **GPU-accelerated**: CUDA JIT-compiled fused kernels with PyTorch fallback
  (34x speedup on RTX 4060).
- **PyTorch-native**: Drop-in ``nn.Module`` layers, standard optimizers, and
  ``torch.compile`` support.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   architecture

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/core
   api/nn
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
