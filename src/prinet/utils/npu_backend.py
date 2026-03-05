"""NPU / DirectML / CPU Backend Abstraction for the Subconscious Controller.

Provides a thin wrapper around ONNX Runtime execution providers so that
the subconscious daemon can transparently run its controller model on
whichever accelerator is available:

    1. **VitisAI EP** — runs directly on the AMD XDNA NPU (requires
       the Ryzen AI SDK custom ONNX Runtime build).
    2. **DirectML EP** — runs on the GPU via DirectML (onnxruntime-directml).
    3. **CPU EP** — always-available fallback.

Environment variables:
    ``RYZEN_AI_INSTALLATION_PATH``
        Root of the Ryzen AI SDK installation
        (e.g. ``C:\\Program Files\\RyzenAI\\1.7.0``).  Firmware, config,
        and cache directories are derived from this.
    ``XLNX_VART_FIRMWARE``
        Path to the NPU firmware binary (e.g. ``1x4.xclbin``).
        Auto-resolved from the SDK installation path when unset.
    ``PRINET_SUBCONSCIOUS_BACKEND``
        Override automatic backend detection.  Accepted values:
        ``npu``, ``directml``, ``cpu``.
    ``PRINET_NPU_TARGET``
        Override the VitisAI compilation target.  Defaults to ``X1``
        (Phoenix / Hawk Point — AIE2).  Set to ``X2`` for Strix Point
        (AIE2P) hardware.

Example:
    >>> backend = detect_best_backend()
    >>> session = create_session("subconscious_controller.onnx", backend)
    >>> print(session.get_providers())
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

try:
    import onnxruntime as ort

    _ORT_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _ORT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

BackendType = Literal["npu", "directml", "cpu"]
"""Execution-provider identifier understood by :func:`create_session`."""

# ---------------------------------------------------------------------------
# Ryzen AI SDK 1.7.0 paths
# ---------------------------------------------------------------------------

_SDK_INSTALL_DIR: str = os.environ.get(
    "RYZEN_AI_INSTALLATION_PATH",
    r"C:\Program Files\RyzenAI\1.7.0",
)
"""Root of the Ryzen AI SDK installation."""

_VOE_DIR: str = os.path.join(_SDK_INSTALL_DIR, "voe-4.0-win_amd64")
"""VitisAI ONNX EP runtime directory (config, xclbins, DLLs)."""

_DEFAULT_XCLBIN_DIR: str = os.path.join(_VOE_DIR, "xclbins", "phoenix")
"""Default xclbin firmware directory for Phoenix / Hawk Point NPU."""

_DEFAULT_XCLBIN: str = "1x4.xclbin"
"""Default xclbin binary — small 1×4 overlay with lowest latency."""

_VITISAI_CONFIG_FILE: str = os.path.join(_VOE_DIR, "vaip_config.json")
"""VitisAI provider option key expected by the Ryzen AI SDK."""

_DEFAULT_TARGET: str = os.environ.get("PRINET_NPU_TARGET", "X1")
"""Compilation target: X1 = Phoenix/Hawk Point (AIE2), X2 = Strix (AIE2P)."""

_CACHE_DIR: str = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "prinet",
    "vitisai_cache",
)
"""Directory for VitisAI compilation cache (speeds up session re-creation)."""

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: backend detection
# ---------------------------------------------------------------------------


def _available_eps() -> list[str]:
    """Return the list of execution providers registered with ONNX Runtime.

    Returns:
        List of EP name strings (e.g. ``['DmlExecutionProvider', ...]``).
        Empty list when ONNX Runtime is not installed.
    """
    if not _ORT_AVAILABLE:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:  # pragma: no cover
        return []


def detect_best_backend() -> BackendType:
    """Auto-detect the most capable ONNX Runtime execution provider.

    Detection order:

    1. Environment variable ``PRINET_SUBCONSCIOUS_BACKEND`` (explicit override).
    2. ``VitisAIExecutionProvider`` (AMD XDNA NPU) if present.
    3. ``DmlExecutionProvider`` (DirectML — GPU or NPU) if present.
    4. ``CPUExecutionProvider`` (always available).

    Returns:
        The best available :data:`BackendType`.
    """
    override = os.environ.get("PRINET_SUBCONSCIOUS_BACKEND", "").strip().lower()
    if override in {"npu", "directml", "cpu"}:
        logger.info("Backend override via env: %s", override)
        return override  # type: ignore[return-value]

    eps = _available_eps()
    if "VitisAIExecutionProvider" in eps:
        logger.info("Detected VitisAI EP — selecting NPU backend.")
        return "npu"
    if "DmlExecutionProvider" in eps:
        logger.info("Detected DirectML EP — selecting DirectML backend.")
        return "directml"

    logger.info("No accelerator EPs found — falling back to CPU.")
    return "cpu"


def npu_available() -> bool:
    """Check whether the VitisAI (NPU) execution provider is present.

    Returns:
        ``True`` if ``VitisAIExecutionProvider`` is registered with ORT.
    """
    return "VitisAIExecutionProvider" in _available_eps()


def directml_available() -> bool:
    """Check whether the DirectML execution provider is present.

    Returns:
        ``True`` if ``DmlExecutionProvider`` is registered with ORT.
    """
    return "DmlExecutionProvider" in _available_eps()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------


def _resolve_firmware_path() -> str:
    """Resolve the NPU firmware binary path.

    Resolution order:

    1. ``XLNX_VART_FIRMWARE`` environment variable (explicit override).
    2. Default xclbin in the Ryzen AI SDK installation:
       ``<SDK>/voe-4.0-win_amd64/xclbins/phoenix/<_DEFAULT_XCLBIN>``.

    Returns:
        Absolute path to the ``.xclbin`` firmware file.

    Raises:
        FileNotFoundError: If no firmware file can be located.
    """
    env_fw = os.environ.get("XLNX_VART_FIRMWARE", "").strip()
    if env_fw and Path(env_fw).is_file():
        return str(Path(env_fw).resolve())

    default = Path(_DEFAULT_XCLBIN_DIR) / _DEFAULT_XCLBIN
    if default.is_file():
        return str(default.resolve())

    msg = (
        "NPU firmware not found.  Set XLNX_VART_FIRMWARE to the path "
        "of a valid .xclbin file, or ensure the Ryzen AI SDK is installed "
        f"at {_SDK_INSTALL_DIR} (set RYZEN_AI_INSTALLATION_PATH if needed)."
    )
    raise FileNotFoundError(msg)


def _build_provider_list(
    backend: BackendType,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build the ``providers`` and ``provider_options`` lists for ORT.

    For the NPU backend the VitisAI EP is configured with:

    * ``config_file`` — path to ``vaip_config.json``
    * ``xclbin`` — path to the NPU firmware binary
    * ``target`` — compilation target (``X1`` for Phoenix/Hawk Point,
      ``X2`` for Strix Point)
    * ``cache_dir`` / ``cache_key`` — VitisAI compilation cache for
      fast session re-creation

    Args:
        backend: Target backend.

    Returns:
        ``(providers, provider_options)`` ready for
        :class:`onnxruntime.InferenceSession`.
    """
    if backend == "npu":
        firmware = _resolve_firmware_path()
        # Ensure cache directory exists
        Path(_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        return (
            ["VitisAIExecutionProvider", "CPUExecutionProvider"],
            [
                {
                    "config_file": _VITISAI_CONFIG_FILE,
                    "xclbin": firmware,
                    "target": _DEFAULT_TARGET,
                    "cache_dir": _CACHE_DIR,
                    "cache_key": "subconscious_v1",
                },
                {},
            ],
        )
    if backend == "directml":
        return (
            ["DmlExecutionProvider", "CPUExecutionProvider"],
            [{}, {}],
        )
    # cpu
    return (["CPUExecutionProvider"], [{}])


def create_session(
    model_path: str | Path,
    backend: BackendType | None = None,
    *,
    inter_op_threads: int = 1,
    intra_op_threads: int = 2,
) -> "ort.InferenceSession":
    """Create an ONNX Runtime inference session on the selected backend.

    If *backend* is ``None``, :func:`detect_best_backend` is called
    automatically.

    Args:
        model_path: Path to the ``.onnx`` model file.
        backend: Execution provider to target.  ``None`` for auto-detect.
        inter_op_threads: Number of inter-op parallelism threads.
        intra_op_threads: Number of intra-op parallelism threads.

    Returns:
        A ready-to-use :class:`onnxruntime.InferenceSession`.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
        FileNotFoundError: If *model_path* does not exist.
    """
    if not _ORT_AVAILABLE:
        msg = (
            "onnxruntime is required for the subconscious controller.  "
            "Install it with:  pip install onnxruntime-directml"
        )
        raise ImportError(msg)

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    if backend is None:
        backend = detect_best_backend()

    providers, provider_options = _build_provider_list(backend)

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = inter_op_threads
    opts.intra_op_num_threads = intra_op_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    logger.info(
        "Creating ORT session: model=%s, backend=%s, providers=%s",
        model_path.name,
        backend,
        providers,
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=providers,
        provider_options=provider_options,
    )
    logger.info("Active providers: %s", session.get_providers())
    return session


# ---------------------------------------------------------------------------
# Convenience: backend info dict (for telemetry / benchmarks)
# ---------------------------------------------------------------------------


def backend_info() -> dict[str, Any]:
    """Return a summary dict of the current backend configuration.

    Useful for embedding in benchmark JSON outputs and session logs.

    Returns:
        Dictionary with keys ``ort_available``, ``ort_version``,
        ``available_eps``, ``best_backend``, ``npu_firmware_found``,
        ``sdk_install_dir``, ``npu_target``, ``cache_dir``.
    """
    eps = _available_eps()
    firmware_found = False
    firmware_path = ""
    try:
        firmware_path = _resolve_firmware_path()
        firmware_found = True
    except FileNotFoundError:
        pass

    ort_version = ""
    if _ORT_AVAILABLE:
        ort_version = getattr(ort, "__version__", "unknown")

    return {
        "ort_available": _ORT_AVAILABLE,
        "ort_version": ort_version,
        "available_eps": eps,
        "best_backend": detect_best_backend(),
        "npu_firmware_found": firmware_found,
        "npu_firmware_path": firmware_path,
        "sdk_install_dir": _SDK_INSTALL_DIR,
        "npu_target": _DEFAULT_TARGET,
        "cache_dir": _CACHE_DIR,
    }
