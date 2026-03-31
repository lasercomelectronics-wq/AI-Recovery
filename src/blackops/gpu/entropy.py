from __future__ import annotations

import numpy as np
import time

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


_GPU_VRAM_POOL_LIMIT_BYTES = int(4.8 * 1024**3)  # 60% VRAM => 4.8GB
_POOL_LIMIT_CONFIGURED = False


def _ensure_cupy_pool_limit() -> None:
    global _POOL_LIMIT_CONFIGURED
    if cp is None or _POOL_LIMIT_CONFIGURED:
        return
    mempool = cp.get_default_memory_pool()
    mempool.set_limit(size=int(_GPU_VRAM_POOL_LIMIT_BYTES))
    _POOL_LIMIT_CONFIGURED = True


def _gpu_duty_cycle_sleep(exec_time_sec: float, max_duty_cycle: float = 0.60) -> None:
    """
    Sleep proportional to kernel execution time so duty cycle <= max_duty_cycle.
    sleep >= exec_time * (1/max_duty_cycle - 1).
    """
    if exec_time_sec <= 0:
        return
    sleep_sec = exec_time_sec * ((1.0 / max_duty_cycle) - 1.0)
    if sleep_sec > 0:
        time.sleep(sleep_sec)


def _entropy_from_histogram(hist: np.ndarray) -> float:
    probs = hist.astype(np.float64)
    total = probs.sum()
    if total == 0:
        return 0.0
    probs /= total
    nz = probs > 0
    return float(-(probs[nz] * np.log2(probs[nz])).sum())


def entropy_cpu(data: bytes | np.ndarray) -> float:
    arr = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else data
    hist = np.bincount(arr, minlength=256)
    return _entropy_from_histogram(hist)


def entropy_cupy(data: bytes | np.ndarray) -> float:
    if cp is None:
        return entropy_cpu(data)

    _ensure_cupy_pool_limit()

    arr = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else data
    darr = cp.asarray(arr, dtype=cp.uint8)
    start = time.perf_counter()
    hist = cp.bincount(darr, minlength=256).astype(cp.float64)
    probs = hist / cp.sum(hist)
    probs = probs[probs > 0]
    ent = -(probs * cp.log2(probs)).sum()
    # Ensure kernels/ops completed before timing and duty-cycle sleep.
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    exec_time_sec = time.perf_counter() - start
    _gpu_duty_cycle_sleep(exec_time_sec)
    return float(cp.asnumpy(ent))


def sliding_entropy_cupy(data: bytes | np.ndarray, window: int = 4096, step: int = 1024) -> np.ndarray:
    """Example CuPy kernel port for entropy windows on GPU."""
    arr = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else data
    if arr.size == 0:
        return np.asarray([0.0], dtype=np.float32)
    if arr.size < window:
        return np.asarray([entropy_cupy(arr)], dtype=np.float32)

    if cp is None:
        values = [entropy_cpu(arr[i : i + window]) for i in range(0, arr.size - window + 1, step)]
        return np.asarray(values, dtype=np.float32)

    _ensure_cupy_pool_limit()

    # CuPy RawKernel example
    kernel = cp.RawKernel(
        r'''
        extern "C" __global__
        void entropy_window(const unsigned char* data, const int size, const int window, const int step, float* out) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            int start = idx * step;
            if (start + window > size) return;

            unsigned int hist[256];
            for (int i = 0; i < 256; ++i) hist[i] = 0;
            for (int i = start; i < start + window; ++i) hist[data[i]]++;

            float e = 0.0f;
            for (int i = 0; i < 256; ++i) {
                if (hist[i] > 0) {
                    float p = (float)hist[i] / (float)window;
                    e -= p * log2f(p);
                }
            }
            out[idx] = e;
        }
        ''',
        "entropy_window",
    )

    darr = cp.asarray(arr, dtype=cp.uint8)
    windows = (arr.size - window) // step + 1
    out = cp.zeros((windows,), dtype=cp.float32)
    threads = 128
    blocks = (windows + threads - 1) // threads
    start = time.perf_counter()
    kernel((blocks,), (threads,), (darr, arr.size, window, step, out))
    # Sync so timing reflects actual kernel runtime.
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    exec_time_sec = time.perf_counter() - start
    _gpu_duty_cycle_sleep(exec_time_sec)
    return cp.asnumpy(out)
