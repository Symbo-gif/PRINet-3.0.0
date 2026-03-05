# PRINet Getting Started Tutorial

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/michaelmaillet/prinet.git PRINet
cd PRINet
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -e ".[dev]"
```

### Verify Installation

```python
import prinet
print(prinet.__version__)  # 3.0.0
print(f"Symbols: {len(prinet.__all__)}")
```

## Tutorial 1: Basic Oscillator Dynamics

Create a Kuramoto oscillator network and observe synchronization:

```python
import torch
from prinet import KuramotoOscillator, OscillatorState, kuramoto_order_parameter

# Create 64 oscillators with random initial phases
N = 64
state = OscillatorState.create_random(N, device="cpu")

# Create oscillator model with moderate coupling
osc = KuramotoOscillator(
    n_oscillators=N,
    coupling_strength=2.0,
    coupling_mode="mean_field",
)

# Integrate for 100 steps, track synchronization
dt = 0.01
for step in range(100):
    state = osc.step(state, dt=dt)
    if step % 20 == 0:
        r = kuramoto_order_parameter(state.phase)
        print(f"Step {step:3d}: order param r = {r:.3f}")
# Expect: r increases from ~0.1 to ~0.8+ as oscillators synchronize
```

## Tutorial 2: Classification with PRINetModel

Use PRINet for a simple classification task:

```python
import torch
from prinet import PRINetModel

# Create model: 64 oscillators, 256-dim input, 10 classes
model = PRINetModel(n_resonances=64, n_dims=256, n_concepts=10)

# Random data (replace with real data)
x = torch.randn(32, 256)  # batch=32, features=256
log_probs = model(x)       # [32, 10] log-probabilities

print(f"Output shape: {log_probs.shape}")
print(f"Sum exp(log_probs): {log_probs.exp().sum(dim=-1).mean():.3f}")  # ≈ 1.0
```

## Tutorial 3: Hybrid Architecture on CLEVR-N

The HybridPRINetV2 combines oscillatory binding with Transformer integration:

```python
import torch
from prinet import HybridPRINetV2

# Create hybrid model
model = HybridPRINetV2(
    n_input=64,          # feature dimension
    n_classes=10,
    d_model=64,          # attention layer dimension
    n_heads=4,
    n_layers=2,          # number of OscAttention + FFN blocks
    n_delta=2,           # slow oscillators
    n_theta=4,           # medium oscillators
    n_gamma=8,           # fast oscillators
    coupling_strength=1.0,
)

# Input: batch of flat features [batch, features]
x = torch.randn(16, 64)
log_probs = model(x)

print(f"Output: {log_probs.shape}")  # [16, 10]

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
labels = torch.randint(0, 10, (16,))
loss = torch.nn.functional.nll_loss(log_probs, labels)
loss.backward()
optimizer.step()
print(f"Loss: {loss.item():.3f}")
```

## Tutorial 4: Multi-Object Tracking with PhaseTracker

PhaseTracker preserves object identity across frames using phase continuity.
It takes per-frame detections and performs cross-frame association via
phase correlation and Hungarian assignment:

```python
import torch
from prinet import PhaseTracker

# detection_dim matches per-detection feature size
tracker = PhaseTracker(
    detection_dim=4,
    n_delta=4,
    n_theta=8,
    n_gamma=16,
)

# Two consecutive frames, each with 3 detections of 4 features
dets_t0 = torch.randn(3, 4)
dets_t1 = torch.randn(3, 4)

# Forward pass: returns (matched_indices, similarity_matrix)
matches, sim = tracker(dets_t0, dets_t1)
print(f"Matches: {matches}")      # e.g. tensor([2, 0, 1])
print(f"Similarity shape: {sim.shape}")  # [3, 3]
```

## Tutorial 5: SCALR Optimizer

The synchronization-conditioned optimizer adapts learning rates based on
oscillator coherence:

```python
import torch
from prinet import PRINetModel, SCALROptimizer

model = PRINetModel(n_resonances=32, n_dims=128, n_concepts=5)

optimizer = SCALROptimizer(
    model.parameters(),
    lr=1e-2,
    r_min=0.1,          # minimum lr fraction when desynchronized
    alpha=1.5,           # synchronization sensitivity exponent
    warmup_steps=50,     # steps before SCALR kicks in
)

# Training loop with synchronization feedback
x = torch.randn(16, 128)
labels = torch.randint(0, 5, (16,))

for epoch in range(10):
    log_probs = model(x)
    loss = torch.nn.functional.nll_loss(log_probs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}: loss={loss.item():.3f}")
```

## Tutorial 6: Hierarchical Delta-Theta-Gamma Dynamics

Multi-rate oscillator dynamics model biological brain rhythms:

```python
import torch
from prinet import DiscreteDeltaThetaGamma

# Create hierarchical oscillator system
dtg = DiscreteDeltaThetaGamma(
    n_delta=2,    # slow: context
    n_theta=4,    # medium: sequence
    n_gamma=8,    # fast: binding
    coupling_strength=1.5,
)

# Initialize random phases and unit amplitudes
n_total = 2 + 4 + 8  # 14 oscillators
batch = 8
phase = torch.rand(batch, n_total) * 2 * 3.14159
amp = torch.ones(batch, n_total)

# Step through time
for t in range(50):
    phase, amp = dtg.step(phase, amp, dt=0.01)

print(f"Final phase shape: {phase.shape}")  # [8, 14]
print(f"Phase range: [{phase.min():.2f}, {phase.max():.2f}]")
```

## Running Tests

```bash
# Full suite
pytest tests/ -v

# Fast tests only (skip slow training tests)
pytest tests/ -v -m "not slow"

# GPU tests
pytest tests/ -v -m gpu

# Specific module
pytest tests/test_core.py -v
```

## Next Steps

- Read the [Architecture Guide](Architecture_Guide.md) for detailed design documentation
- See [API Reference: Coupling Topologies](API_Reference_Coupling_Topologies.md)
  for oscillator coupling modes
- Explore `benchmarks/` for performance evaluation scripts
- Check `Docs/PRINet_Initial_Research_and_foundations/` for mathematical foundations
