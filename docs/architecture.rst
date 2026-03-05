Architecture Overview
=====================

PRINet's architecture is organized into three packages, each containing
specialised modules for oscillatory neural network processing.

Package Structure
-----------------

::

   prinet/
   +-- core/         # Fundamental dynamics
   |   +-- decomposition   # CP tensor decomposition
   |   +-- measurement     # Order parameters, coherence
   |   +-- subconscious    # Background processing daemon
   |   +-- propagation/    # Oscillator models, coupling, integration
   |       +-- oscillator_models   # Kuramoto, Hopf, DeltaThetaGamma
   |       +-- oscillator_state    # OscillatorState dataclass
   |       +-- coupling            # Mean-field, sparse k-NN, small-world
   |       +-- integrators         # RK4, RK45, exponential, multi-rate
   |       +-- inhibition          # Feedforward/feedback inhibition
   |       +-- networks            # DeltaThetaGammaNetwork
   |       +-- temporal            # TemporalSequenceProcessor
   |       +-- sweep_utils         # Phase-space sweep helpers
   +-- nn/           # PyTorch layers and models
   |   +-- layers            # PRINetLayer, ResonanceLayer
   |   +-- activations       # PhaseActivation, ResonanceGating
   |   +-- optimizers        # ScaLR adaptive optimizer
   |   +-- hep               # Holomorphic Error Propagation
   |   +-- hybrid            # HybridPRINetV2 (Oscillator + Transformer)
   |   +-- slot_attention    # Slot attention integration
   |   +-- subconscious_model  # SubconsciousPRINet
   |   +-- adaptive_allocation # Region-aware allocation
   |   +-- training_hooks    # Gradient monitoring, live dashboards
   |   +-- ablation_variants # Ablation study model variants
   +-- utils/        # GPU kernels, solvers, tools
       +-- fused_kernels     # CUDA JIT fused discrete step
       +-- triton_kernels    # Triton GPU kernels + PyTorch fallback
       +-- cuda_kernels      # Low-level CUDA helpers
       +-- oscillosim        # OscilloSim interactive simulator
       +-- datasets          # CLEVR-N, MNIST-N data loaders
       +-- profiler          # PRINetProfiler instrumentation
       +-- temporal_metrics  # Phase-locking value, coherence
       +-- temporal_training # TemporalTrainer curriculum
       +-- benchmark_reporting  # JSON benchmark reporting
       +-- figure_generation # NeurIPS figure generators
       +-- table_generation  # NeurIPS LaTeX table generators

Processing Pipeline
-------------------

1. **Input projection** -- Raw features are projected into oscillator
   amplitudes via a learned linear layer.

2. **Phase initialization** -- Initial phases are drawn from the natural
   frequency distribution or set to a constant.

3. **Coupled integration** -- ``n_layers`` integration steps evolve the
   oscillators through Kuramoto-style coupling.  Each step computes:

   .. math::

      \\dot{\\theta}_i = \\omega_i + \\frac{K}{N}\\sum_{j}
         a_j \\sin(\\theta_j - \\theta_i)

4. **Measurement** -- The Kuramoto order parameter
   :math:`r = |\\frac{1}{N}\\sum_j e^{i\\theta_j}|` is computed per-band.

5. **Readout** -- A learned projection maps the final amplitudes or order
   parameters to class logits.

Multi-Rate Integration
----------------------

The :class:`~prinet.core.DeltaThetaGammaNetwork` uses three frequency
bands operating at different timescales:

- **Delta** (1--4 Hz): slow contextual modulation
- **Theta** (4--8 Hz): sequential binding and working memory
- **Gamma** (30--100 Hz): fast feature binding

The multi-rate RK4 integrator sub-steps faster bands while advancing
slower bands at their native rate, preserving cross-frequency coupling.

CUDA Acceleration
-----------------

The fused discrete-step kernel (``fused_kernels.py``) JIT-compiles via
``torch.utils.cpp_extension.load_inline``, providing a **34x speedup**
over the pure-PyTorch fallback on an RTX 4060.  The build uses
minimal ATen includes to avoid MSVC compatibility issues with
``compiled_autograd.h``.
