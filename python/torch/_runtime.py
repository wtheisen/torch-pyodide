from __future__ import annotations

from typing import Any


RUNTIME_GLOBAL_KEY = "__torch_pyodide_runtime__"


def _to_python(value: Any) -> Any:
    if hasattr(value, "to_py"):
        return value.to_py()
    return value


BACKEND_GLOBAL_KEY = "__torch_pyodide_backend__"
_RUNTIME: Any = None


def _choose_runtime() -> Any:
    """ai2ai: pick the backend once.

    "cpu" (numpy) is used when asked for (globalThis.__torch_pyodide_backend__ or the
    TORCH_PYODIDE_BACKEND environment variable), outside Pyodide, when no WebGPU runtime
    is installed, and by default: the cpu backend matches PyTorch more closely than the
    WebGPU kernels. "webgpu" requires the JS runtime. "auto" uses WebGPU when the runtime
    is installed and the browser has a GPU adapter.
    """
    import os
    from ._numpy_runtime import get_numpy_runtime

    wanted = os.environ.get("TORCH_PYODIDE_BACKEND", "").lower()
    try:
        from js import globalThis  # type: ignore
    except Exception:
        return get_numpy_runtime()
    wanted = str(getattr(globalThis, BACKEND_GLOBAL_KEY, None) or wanted or "cpu").lower()
    runtime = getattr(globalThis, RUNTIME_GLOBAL_KEY, None)
    if wanted == "cpu":
        return get_numpy_runtime()
    if runtime is None:
        if wanted == "webgpu":
            raise RuntimeError(
                "WebGPU backend requested but the torch runtime bridge is not installed "
                "(globalThis.__torch_pyodide_runtime__)."
            )
        return get_numpy_runtime()
    if wanted == "auto":
        nav = getattr(globalThis, "navigator", None)
        if nav is None or getattr(nav, "gpu", None) is None:
            return get_numpy_runtime()
    return runtime


def _get_runtime() -> Any:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _choose_runtime()
    return _RUNTIME


def backend() -> str:
    """"cpu" or "webgpu"."""
    return "cpu" if getattr(_get_runtime(), "backend", None) == "cpu" else "webgpu"


def _run_js_awaitable(awaitable: Any) -> Any:
    if not hasattr(awaitable, "then"):  # cpu backend: already a plain value
        return _to_python(awaitable)
    from pyodide.ffi import can_run_sync, run_sync  # type: ignore

    if not can_run_sync():
        raise RuntimeError(
            "Cannot use synchronous torch API: run_sync is unavailable. "
            "Execute Python through pyodide.runPythonAsync(...) and call "
            "sync Python functions from JS via callPromising(...)."
        )
    return _to_python(run_sync(awaitable))

