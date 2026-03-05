Getting Started
===============

Installation
------------

.. code-block:: bash

   # Clone and install in development mode
   git clone https://github.com/michaelmaillet/prinet.git
   cd prinet
   python -m venv .venv
   .venv\Scripts\activate       # Windows
   # source .venv/bin/activate  # Linux/macOS
   pip install -e ".[dev]"

Verify the installation:

.. code-block:: python

   import prinet
   print(prinet.__version__)
   print(f"Public symbols: {len(prinet.__all__)}")

Tutorial 1: Oscillator Dynamics
-------------------------------

Create a Kuramoto oscillator network and observe synchronization:

.. code-block:: python

   import torch
   from prinet import KuramotoOscillator, OscillatorState, kuramoto_order_parameter

   N = 64
   state = OscillatorState.create_random(N, device="cpu")
   osc = KuramotoOscillator(
       n_oscillators=N,
       coupling_strength=2.0,
       coupling_mode="mean_field",
   )

   dt = 0.01
   for step in range(100):
       state = osc.step(state, dt=dt)
       if step % 20 == 0:
           r = kuramoto_order_parameter(state.phase)
           print(f"Step {step:3d}: r = {r:.3f}")

Tutorial 2: Classification with PRINetModel
--------------------------------------------

.. code-block:: python

   import torch
   from prinet import PRINetModel

   model = PRINetModel(n_resonances=64, n_dims=256, n_concepts=10)
   x = torch.randn(32, 256)
   log_probs = model(x)  # [32, 10]

Tutorial 3: Hybrid Architecture
-------------------------------

The :class:`~prinet.nn.HybridPRINetV2` combines oscillatory binding with
a Transformer encoder:

.. code-block:: python

   import torch
   from prinet import HybridPRINetV2

   model = HybridPRINetV2(
       n_input=512, n_classes=10,
       n_delta=4, n_theta=8, n_gamma=32,
       d_model=64, n_heads=4, n_layers=4,
   )
   x = torch.randn(16, 512)  # (batch, features)
   out = model(x)  # [16, 10]

Tutorial 4: OscilloSim Exploration
-----------------------------------

Use the interactive oscillator simulator:

.. code-block:: python

   from prinet.utils.oscillosim import OscilloSim

   sim = OscilloSim(n_oscillators=128, coupling_strength=1.5)
   result = sim.run(n_steps=500, dt=0.01)
   print(f"Final order parameter: {result.order_parameter[-1]:.4f}")
