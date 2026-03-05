# models/

Pre-trained model weights for the PRINet SubconsciousController.

## Contents

| File | Description |
|---|---|
| `subconscious_controller.onnx` | ONNX-exported SubconsciousController model |
| `subconscious_controller.onnx.data` | External data tensor for the ONNX model |

## Usage

```python
from prinet.core import SubconsciousDaemon
from prinet.utils import create_session

# Load with ONNX Runtime
session = create_session("models/subconscious_controller.onnx")

# Or use the high-level daemon interface
daemon = SubconsciousDaemon(model_path="models/subconscious_controller.onnx")
```

## Requirements

Loading the ONNX model requires the `onnx` optional dependency:

```bash
pip install -e ".[onnx]"
```

On Windows, this includes `onnxruntime-directml` for NPU/GPU acceleration.

## License

MIT — see [LICENSE](../LICENSE).
