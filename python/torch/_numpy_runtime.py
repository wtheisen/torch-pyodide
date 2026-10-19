"""CPU backend for torch-pyodide, written in numpy (ai2ai fork).

It implements the same runtime interface as the JavaScript/WebGPU runtime
(runtime/src/runtime.ts), so the Python layer (autograd, nn, optim) is unchanged.
Tensors live in a dict of numpy arrays keyed by integer id. Every method returns
plain Python values (a {"id", "shape", "dtype"} dict for a tensor), never an
awaitable, and _runtime._run_js_awaitable passes such values straight through.

Semantics follow the WGSL kernels, including their quirks, so that both backends
give the same answers for the Python layer. Float math is done in float32, or float64 for float64 tensors.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

_NP = {
    "float32": np.float32, "float16": np.float32, "bfloat16": np.float32, "float64": np.float64,
    "int8": np.int32, "int16": np.int32, "int32": np.int32, "int64": np.int64,
    "uint8": np.int32, "uint16": np.int32, "uint32": np.int64, "uint64": np.int64,
    "bool": np.bool_,
}


def _npdt(dtype: str | None):
    return _NP.get(str(dtype or "float32").replace("torch.", ""), np.float32)


def _is_float(dtype: str) -> bool:
    return dtype in ("float32", "float16", "bfloat16", "float64")


def _ax(dim: int, rank: int) -> int:
    return dim + rank if dim < 0 else dim


def _f32(x) -> np.ndarray:
    a = np.asarray(x)
    return a if a.dtype == np.float64 else a.astype(np.float32)  # float64 stays float64


def _F(a) -> np.ndarray:
    """float32 math, except float64 arrays stay float64."""
    a = np.asarray(a)
    return a if a.dtype == np.float64 else a.astype(np.float32)


def _erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz-Stegun 7.1.26 is too coarse; use math.erf elementwise (small tensors)
    return np.vectorize(math.erf, otypes=[np.float64])(x).astype(np.float32)


class _H(dict):
    """A tensor handle: works as h["id"] and as h.id, like the JS objects."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


class NumpyRuntime:
    backend = "cpu"

    def __init__(self) -> None:
        self._t: dict[int, np.ndarray] = {}
        self._dt: dict[int, str] = {}
        self._next = 1
        self._rng = np.random.default_rng()

    # ── registry ──────────────────────────────────────────────────────────
    def _reg(self, arr: Any, dtype: str, shape: Sequence[int] | None = None) -> dict:
        dtype = dtype.encode().decode().replace("torch.", "") if isinstance(dtype, str) else str(dtype)
        a = np.array(arr, dtype=_npdt(dtype), copy=True)
        if shape is not None:
            a = a.reshape([int(s) for s in shape])
        tid = self._next
        self._next += 1
        self._t[tid] = a
        self._dt[tid] = dtype
        return _H(id=tid, shape=list(a.shape), dtype=dtype)

    def _g(self, tid: int) -> np.ndarray:
        try:
            return self._t[int(tid)]
        except KeyError:
            raise RuntimeError(f"Unknown tensor id: {tid}.") from None

    def _d(self, tid: int) -> str:
        return self._dt[int(tid)]

    def _like(self, arr, tid, dtype=None):
        return self._reg(arr, dtype or self._d(tid))

    def _put(self, tid: int, arr) -> None:  # in-place write (optimizer steps)
        a = self._t[int(tid)]
        self._t[int(tid)] = np.asarray(arr, dtype=a.dtype).reshape(a.shape)

    # ── device / bookkeeping ──────────────────────────────────────────────
    def init(self, *a): return None
    def isAvailable(self): return False
    def isInitialized(self): return True
    def deviceCount(self): return 0
    def currentDevice(self): return 0
    def getDeviceName(self, *a): return "cpu (numpy)"
    def getDeviceProperties(self, *a): return {"name": "cpu (numpy)", "total_memory": 0}
    def memoryAllocated(self, *a): return int(sum(a.nbytes for a in self._t.values()))
    def memoryReserved(self, *a): return self.memoryAllocated()
    def beginFrame(self): return None
    def endFrame(self): return None
    def cancelFrame(self): return None

    def destroy(self, tid):
        self._t.pop(int(tid), None)
        self._dt.pop(int(tid), None)

    def toList(self, tid):
        a = self._g(tid)
        if a.dtype == np.bool_:
            return [1 if v else 0 for v in a.reshape(-1).tolist()]
        return a.reshape(-1).tolist()

    # ── creation ──────────────────────────────────────────────────────────
    def tensorFromData(self, data, shape, dtype):
        data = list(data.to_py() if hasattr(data, "to_py") else data)
        return self._reg(data, dtype, list(shape))

    def zeros(self, shape, dtype): return self._reg(np.zeros(list(shape)), dtype)
    def ones(self, shape, dtype): return self._reg(np.ones(list(shape)), dtype)
    def empty(self, shape, dtype): return self.zeros(shape, dtype)
    def full(self, shape, value, dtype): return self._reg(np.full(list(shape), value), dtype)

    def _likeshape(self, tid, value, dtype):
        a = self._g(tid)
        return self._reg(np.full(a.shape, value), dtype or self._d(tid))

    def zerosLike(self, tid, dtype=None): return self._likeshape(tid, 0, dtype)
    def onesLike(self, tid, dtype=None): return self._likeshape(tid, 1, dtype)
    def emptyLike(self, tid, dtype=None): return self._likeshape(tid, 0, dtype)
    def fullLike(self, tid, value, dtype=None): return self._likeshape(tid, value, dtype)

    def setSeed(self, seed): self._rng = np.random.default_rng(int(seed))
    def rand(self, shape, dtype): return self._reg(self._rng.random(list(shape)), dtype)
    def randn(self, shape, dtype): return self._reg(self._rng.standard_normal(list(shape)), dtype)
    def normal(self, shape, dtype, mean, std): return self._reg(self._rng.normal(mean, std, list(shape)), dtype)
    def bernoulli(self, shape, dtype, p): return self._reg(self._rng.random(list(shape)) < p, dtype)
    def exponential(self, shape, dtype, rate): return self._reg(self._rng.exponential(1.0 / rate, list(shape)), dtype)
    def logNormal(self, shape, dtype, mean, std): return self._reg(self._rng.lognormal(mean, std, list(shape)), dtype)

    def arange(self, start, end, step, dtype):
        return self._reg(np.arange(start, end, step, dtype=np.float64), dtype)

    def fill(self, tid, value): return self._likeshape(tid, value, None)

    def cast(self, tid, dtype):
        x = self._g(tid)
        if dtype == "bool":
            return self._reg(x != 0, "bool")
        if not _is_float(dtype) and x.dtype.kind == "f":
            x = np.trunc(x)
        return self._reg(x, dtype)

    # ── binary arithmetic (result dtype = first operand's, as in the kernels) ──
    def _promote(self, a, b) -> str:
        """PyTorch type promotion (the JS kernels just keep the first operand's dtype)."""
        da, db = self._d(a), self._d(b)
        if "float64" in (da, db):
            return "float64"
        if _is_float(da) or _is_float(db):
            return "float32"
        if da == "bool" and db == "bool":
            return "bool"
        return "int64" if "int64" in (da, db) else (da if da != "bool" else db)

    def _bin(self, a, b, fn, out_dtype=None):
        x, y = self._g(a), self._g(b)
        dt = out_dtype or self._promote(a, b)
        with np.errstate(all="ignore"):
            if _is_float(dt):
                r = fn(x.astype(_npdt(dt)), y.astype(_npdt(dt)))
            else:
                r = fn(x, y)
        return self._reg(r, dt)

    def add(self, a, b): return self._bin(a, b, np.add)
    def sub(self, a, b): return self._bin(a, b, np.subtract)
    def mul(self, a, b): return self._bin(a, b, np.multiply)
    def div(self, a, b): return self._bin(a, b, np.true_divide, "float32" if not _is_float(self._promote(a, b)) else None)
    def trueDivide(self, a, b): return self.div(a, b)
    def floorDivide(self, a, b): return self._bin(a, b, lambda x, y: np.floor(np.true_divide(x, y)))
    def pow(self, a, b): return self._bin(a, b, np.power)
    def atan2(self, a, b): return self._bin(a, b, np.arctan2)
    def hypot(self, a, b): return self._bin(a, b, np.hypot)
    def logaddexp(self, a, b): return self._bin(a, b, np.logaddexp)
    def logaddexp2(self, a, b): return self._bin(a, b, np.logaddexp2)
    def fmod(self, a, b): return self._bin(a, b, np.fmod)
    def remainder(self, a, b): return self._bin(a, b, lambda x, y: x - np.floor(x / y) * y)
    def copysign(self, a, b): return self._bin(a, b, np.copysign)
    def nextafter(self, a, b): return self._bin(a, b, np.nextafter)
    def heaviside(self, a, b): return self._bin(a, b, np.heaviside)
    def maximum(self, a, b): return self._bin(a, b, np.maximum)
    def minimum(self, a, b): return self._bin(a, b, np.minimum)

    def xlogy(self, a, b):
        return self._bin(a, b, lambda x, y: np.where(x == 0, 0.0, x * np.log(y)))

    def logicalAnd(self, a, b): return self._bin(a, b, np.logical_and, "bool")
    def logicalOr(self, a, b): return self._bin(a, b, np.logical_or, "bool")
    def logicalXor(self, a, b): return self._bin(a, b, np.logical_xor, "bool")

    def _bits(self, a, b, fn):
        x, y = self._g(a), self._g(b)
        r = fn(x.astype(np.int64), y.astype(np.int64))
        return self._reg(r if self._d(a) != "bool" else r.astype(bool), self._d(a))

    def bitwiseAnd(self, a, b): return self._bits(a, b, np.bitwise_and)
    def bitwiseOr(self, a, b): return self._bits(a, b, np.bitwise_or)
    def bitwiseXor(self, a, b): return self._bits(a, b, np.bitwise_xor)

    def bitwiseNot(self, a):
        x = self._g(a)
        return self._like(~x if x.dtype == np.bool_ else ~x.astype(np.int64), a)

    def mulScalar(self, tid, value):
        x = self._g(tid)
        return self._like(_F(x) * np.float32(value) if _is_float(self._d(tid)) else x * value, tid)

    def clamp(self, tid, lo, hi):
        x = self._g(tid)
        return self._like(np.clip(x, -np.inf if lo is None else lo, np.inf if hi is None else hi), tid)

    def where(self, c, x, y):
        cc, xx, yy = self._g(c), self._g(x), self._g(y)
        return self._reg(np.where(cc.astype(bool), xx, yy), self._d(x))

    def lerpScalar(self, s, e, w):
        a, b = self._g(s), self._g(e)
        return self._like(a + np.float32(w) * (b - a), s)

    def lerpTensor(self, s, e, w):
        a, b, ww = self._g(s), self._g(e), self._g(w)
        return self._like(a + ww * (b - a), s)

    def addcmul(self, i, t1, t2, value):
        return self._like(self._g(i) + np.float32(value) * self._g(t1) * self._g(t2), i)

    def addcdiv(self, i, t1, t2, value):
        return self._like(self._g(i) + np.float32(value) * self._g(t1) / self._g(t2), i)

    def matmul(self, a, b):
        x, y = _F(self._g(a)), _F(self._g(b))
        return self._reg(np.matmul(x, y), self._d(a) if _is_float(self._d(a)) else "float32")

    mm = matmul
    bmm = matmul
    mv = matmul

    # ── comparisons ───────────────────────────────────────────────────────
    def eq(self, a, b): return self._bin(a, b, np.equal, "bool")
    def ne(self, a, b): return self._bin(a, b, np.not_equal, "bool")
    def lt(self, a, b): return self._bin(a, b, np.less, "bool")
    def le(self, a, b): return self._bin(a, b, np.less_equal, "bool")
    def gt(self, a, b): return self._bin(a, b, np.greater, "bool")
    def ge(self, a, b): return self._bin(a, b, np.greater_equal, "bool")

    # ── unary ─────────────────────────────────────────────────────────────
    def _un(self, tid, fn, out_dtype=None):
        x = self._g(tid)
        with np.errstate(all="ignore"):
            r = fn(x.astype(np.float64 if x.dtype == np.float64 else np.float32) if x.dtype != np.bool_ or out_dtype != "bool" else x)
        return self._reg(r, out_dtype or self._d(tid))

    def relu(self, t): return self._un(t, lambda x: np.maximum(x, 0))
    def abs(self, t): return self._un(t, np.abs)
    def sqrt(self, t): return self._un(t, np.sqrt)
    def rsqrt(self, t): return self._un(t, lambda x: 1 / np.sqrt(x))
    def exp(self, t): return self._un(t, np.exp)
    def exp2(self, t): return self._un(t, np.exp2)
    def expm1(self, t): return self._un(t, np.expm1)
    def log(self, t): return self._un(t, np.log)
    def log2(self, t): return self._un(t, np.log2)
    def log10(self, t): return self._un(t, np.log10)
    def log1p(self, t): return self._un(t, np.log1p)
    def neg(self, t): return self._un(t, np.negative)
    def sigmoid(self, t): return self._un(t, lambda x: 1 / (1 + np.exp(-x)))
    def tanh(self, t): return self._un(t, np.tanh)
    def sin(self, t): return self._un(t, np.sin)
    def cos(self, t): return self._un(t, np.cos)
    def tan(self, t): return self._un(t, np.tan)
    def asin(self, t): return self._un(t, np.arcsin)
    def acos(self, t): return self._un(t, np.arccos)
    def atan(self, t): return self._un(t, np.arctan)
    def sinh(self, t): return self._un(t, np.sinh)
    def cosh(self, t): return self._un(t, np.cosh)
    def asinh(self, t): return self._un(t, np.arcsinh)
    def acosh(self, t): return self._un(t, np.arccosh)
    def atanh(self, t): return self._un(t, np.arctanh)
    def floor(self, t): return self._un(t, np.floor)
    def ceil(self, t): return self._un(t, np.ceil)
    def round(self, t): return self._un(t, np.round)
    def trunc(self, t): return self._un(t, np.trunc)
    def frac(self, t): return self._un(t, lambda x: x - np.trunc(x))
    def reciprocal(self, t): return self._un(t, lambda x: 1 / x)
    def square(self, t): return self._un(t, np.square)
    def sign(self, t): return self._un(t, np.sign)
    def sgn(self, t): return self._un(t, np.sign)
    def softplus(self, t): return self._un(t, lambda x: np.logaddexp(0, x))
    def mish(self, t): return self._un(t, lambda x: x * np.tanh(np.logaddexp(0, x)))
    def silu(self, t): return self._un(t, lambda x: x / (1 + np.exp(-x)))
    def hardsigmoid(self, t): return self._un(t, lambda x: np.clip(x / 6 + 0.5, 0, 1))
    def hardswish(self, t): return self._un(t, lambda x: x * np.clip(x / 6 + 0.5, 0, 1))
    def softsign(self, t): return self._un(t, lambda x: x / (1 + np.abs(x)))
    def tanhshrink(self, t): return self._un(t, lambda x: x - np.tanh(x))
    def deg2rad(self, t): return self._un(t, np.deg2rad)
    def rad2deg(self, t): return self._un(t, np.rad2deg)
    def erf(self, t): return self._un(t, _erf)
    def erfc(self, t): return self._un(t, lambda x: 1 - _erf(x))
    def lgamma(self, t): return self._un(t, np.vectorize(math.lgamma, otypes=[np.float64]))
    def i0(self, t): return self._un(t, np.i0)

    def digamma(self, t):
        def dg(v):
            r = 0.0
            while v < 6:
                r -= 1 / v
                v += 1
            f = 1 / (v * v)
            return r + math.log(v) - 0.5 / v - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))
        return self._un(t, np.vectorize(dg, otypes=[np.float64]))

    def gelu(self, t):  # exact (erf) GELU, PyTorch's default
        return self._un(t, lambda x: 0.5 * x * (1 + _erf(x / np.float32(math.sqrt(2)))))

    def leakyRelu(self, t, alpha=0.01): return self._un(t, lambda x: np.where(x > 0, x, x * np.float32(alpha)))

    def isnan(self, t): return self._un(t, np.isnan, "bool")
    def isinf(self, t): return self._un(t, np.isinf, "bool")
    def isfinite(self, t): return self._un(t, np.isfinite, "bool")
    def isposinf(self, t): return self._un(t, np.isposinf, "bool")
    def isneginf(self, t): return self._un(t, np.isneginf, "bool")
    def logicalNot(self, t): return self._reg(~self._g(t).astype(bool), "bool")

    # ── reductions ────────────────────────────────────────────────────────
    def _red(self, tid, fn, dtype=None):
        x = self._g(tid)
        return self._reg(np.asarray(fn(_F(x) if x.dtype != np.bool_ else x)), dtype or self._d(tid))

    def sum(self, t): return self._red(t, np.sum, None if _is_float(self._d(t)) else "int64")
    def mean(self, t): return self._red(t, np.mean)
    def prod(self, t): return self._red(t, np.prod, None if _is_float(self._d(t)) else "int64")
    def min(self, t): return self._red(t, np.min)
    def max(self, t): return self._red(t, np.max)
    def any(self, t): return self._reg(np.asarray(np.any(self._g(t))), "bool")
    def all(self, t): return self._reg(np.asarray(np.all(self._g(t))), "bool")
    def argmax(self, t): return self._reg(np.asarray(np.argmax(self._g(t))), "int32")
    def argmin(self, t): return self._reg(np.asarray(np.argmin(self._g(t))), "int32")
    def cumsum(self, t): return self._like(np.cumsum(self._g(t).reshape(-1)).reshape(self._g(t).shape), t,
                                          None if _is_float(self._d(t)) else "int64")
    def cumprod(self, t): return self._like(np.cumprod(self._g(t).reshape(-1)).reshape(self._g(t).shape), t)

    def sumDim(self, t, dim, keepdim):
        x = self._g(t)
        return self._like(np.sum(_F(x) if _is_float(self._d(t)) else x, axis=_ax(dim, x.ndim), keepdims=bool(keepdim)), t,
                          None if _is_float(self._d(t)) else "int64")

    def meanDim(self, t, dim, keepdim):
        x = _F(self._g(t))
        return self._like(np.mean(x, axis=_ax(dim, x.ndim), keepdims=bool(keepdim)), t)

    def softmax(self, t, dim):
        x = _F(self._g(t))
        ax = _ax(dim, x.ndim)
        e = np.exp(x - np.max(x, axis=ax, keepdims=True))
        return self._like(e / np.sum(e, axis=ax, keepdims=True), t)

    def logSoftmax(self, t, dim):
        x = _F(self._g(t))
        ax = _ax(dim, x.ndim)
        m = np.max(x, axis=ax, keepdims=True)
        return self._like(x - m - np.log(np.sum(np.exp(x - m), axis=ax, keepdims=True)), t)

    def _rows(self, inp, tgt):
        x = _F(self._g(inp))
        y = self._g(tgt).reshape(-1).astype(np.int64)
        return x.reshape(len(y), -1), y

    def nllLoss(self, inp, tgt):
        x, y = self._rows(inp, tgt)
        return self._reg(-x[np.arange(len(y)), y], self._d(inp))

    def nllLossReduced(self, inp, tgt, reduction):
        x, y = self._rows(inp, tgt)
        v = -x[np.arange(len(y)), y]
        return self._reg(np.asarray(v.sum() if reduction == "sum" else v.mean()), self._d(inp))

    def nllLossBackward(self, tgt, batch, classes, scale=1.0):
        y = self._g(tgt).reshape(-1).astype(np.int64)
        g = np.zeros((int(batch), int(classes)), np.float32)
        g[np.arange(int(batch)), y[: int(batch)]] = -np.float32(scale)
        return self._reg(g, "float32")

    def crossEntropy(self, inp, tgt):
        x, y = self._rows(inp, tgt)
        m = x.max(axis=1, keepdims=True)
        lse = (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))[:, 0]
        return self._reg(lse - x[np.arange(len(y)), y], self._d(inp))

    def crossEntropyBackward(self, gout, inp, tgt, reduction):
        x, y = self._rows(inp, tgt)
        b = len(y)
        e = np.exp(x - x.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        oh = np.zeros_like(p)
        oh[np.arange(b), np.clip(y, 0, x.shape[1] - 1)] = 1
        go = _F(self._g(gout)).reshape(-1)
        scale = 1.0 / max(1, b) if reduction == "mean" else 1.0
        g = (p - oh) * (go[0] if go.size == 1 else go.reshape(b, 1)) * np.float32(scale)
        return self._reg(g, self._d(inp))

    def logSoftmaxBackward(self, gout, sm, batch, classes):
        g = _F(self._g(gout)).reshape(int(batch), int(classes))
        s = _F(self._g(sm)).reshape(int(batch), int(classes))
        return self._reg(g - s * g.sum(axis=1, keepdims=True), self._d(gout))

    def softmaxBackward(self, gout, sm, batch, classes):
        g = _F(self._g(gout)).reshape(int(batch), int(classes))
        s = _F(self._g(sm)).reshape(int(batch), int(classes))
        return self._reg(s * (g - (g * s).sum(axis=1, keepdims=True)), self._d(gout))

    def maxMinBackward(self, inp, gout, mode):
        x = self._g(inp).reshape(-1)
        g = np.zeros(x.shape, np.float32)
        if x.size:
            g[int(np.argmax(x) if mode == "max" else np.argmin(x))] = float(self._g(gout).reshape(-1)[0])
        return self._reg(g.reshape(self._g(inp).shape), self._d(inp))

    def pdist(self, t, p):
        x = _F(self._g(t))
        n = x.shape[0]
        out = [np.sum(np.abs(x[i] - x[j]) ** p) ** (1.0 / p) for i in range(n) for j in range(i + 1, n)]
        return self._reg(np.asarray(out), self._d(t))

    # ── optimizer steps (in place, formulas copied from the WGSL kernels) ──
    def sgdStep(self, p, g, buf, lr, momentum, wd, dampening, nesterov):
        P, G = _F(self._g(p)), _F(self._g(g))
        if wd:
            G = G + np.float32(wd) * P
        upd = G
        if momentum:
            B = _F(self._g(buf)) * np.float32(momentum) + G * np.float32(1 - dampening)
            self._put(buf, B)
            upd = G + np.float32(momentum) * B if nesterov else B
        self._put(p, P - np.float32(lr) * upd)

    def adamStep(self, p, g, m, v, lr, b1, b2, eps, wd, step_size, inv_sqrt_bc2):
        P, G = _F(self._g(p)), _F(self._g(g))
        if wd:
            G = G + np.float32(wd) * P
        M = self._g(m) * np.float32(b1) + np.float32(1 - b1) * G
        V = self._g(v) * np.float32(b2) + np.float32(1 - b2) * G * G
        self._put(m, M)
        self._put(v, V)
        self._put(p, P - (M / (np.sqrt(V) * np.float32(inv_sqrt_bc2) + np.float32(eps))) * np.float32(step_size))

    def adamWStep(self, p, g, m, v, lr, b1, b2, eps, wd, step_size, inv_sqrt_bc2):
        P = _F(self._g(p))
        if wd:
            P = P * np.float32(1 - lr * wd)
            self._put(p, P)
        self.adamStep(p, g, m, v, lr, b1, b2, eps, 0.0, step_size, inv_sqrt_bc2)

    def rmspropStep(self, p, g, sq, buf, lr, alpha, eps, wd, momentum):
        P, G = _F(self._g(p)), _F(self._g(g))
        if wd:
            G = G + np.float32(wd) * P
        S = self._g(sq) * np.float32(alpha) + np.float32(1 - alpha) * G * G
        self._put(sq, S)
        upd = G / (np.sqrt(S) + np.float32(eps))
        if momentum:
            B = self._g(buf) * np.float32(momentum) + upd
            self._put(buf, B)
            upd = B
        self._put(p, P - np.float32(lr) * upd)

    def adagradStep(self, p, g, ss, lr, eps, wd):
        P, G = _F(self._g(p)), _F(self._g(g))
        if wd:
            G = G + np.float32(wd) * P
        S = self._g(ss) + G * G
        self._put(ss, S)
        self._put(p, P - np.float32(lr) * G / (np.sqrt(S) + np.float32(eps)))

    def adamaxStep(self, p, g, m, u, lr, b1, b2, eps, wd, step_size, bc1):
        P, G = _F(self._g(p)), _F(self._g(g))
        if wd:
            G = G + np.float32(wd) * P
        M = self._g(m) * np.float32(b1) + np.float32(1 - b1) * G
        U = np.maximum(self._g(u) * np.float32(b2), np.abs(G))
        self._put(m, M)
        self._put(u, U)
        self._put(p, P - (M / (U + np.float32(eps))) * np.float32(step_size) * np.float32(bc1))

    def _mv(self, p, g, m, v, b1, b2, wd):
        P, G = _F(self._g(p)), _F(self._g(g))
        if wd:
            G = G + np.float32(wd) * P
        M = self._g(m) * np.float32(b1) + np.float32(1 - b1) * G
        V = self._g(v) * np.float32(b2) + np.float32(1 - b2) * G * G
        self._put(m, M)
        self._put(v, V)
        return P, G, M, V

    def nadamStep(self, p, g, m, v, lr, b1, b2, eps, wd, step_size, mu):
        P, G, M, V = self._mv(p, g, m, v, b1, b2, wd)
        mh = np.float32(mu) * M + np.float32(1 - mu) * G
        self._put(p, P - (mh / (np.sqrt(V) + np.float32(eps))) * np.float32(step_size))

    def radamStep(self, p, g, m, v, lr, b1, b2, eps, wd, step_size, b1t, b2t, step_count=1):
        P, G, M, V = self._mv(p, g, m, v, b1, b2, wd)
        mh = M / np.float32(1 - b1t)
        t = max(float(step_count), 1.0)
        rho_inf = 2.0 / (1.0 - b2) - 1.0
        rho_t = rho_inf - 2.0 * t * b2t / max(1.0 - b2t, 1e-10)
        if rho_t > 5.0:
            rect = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            upd = np.float32(step_size) * mh * np.float32(rect)
        else:
            upd = np.float32(step_size) * mh
        self._put(p, P - upd)

    # ── masking ───────────────────────────────────────────────────────────
    def maskedSelect(self, t, mask):
        x = self._g(t)
        return self._like(x.reshape(-1)[np.broadcast_to(self._g(mask).astype(bool), x.shape).reshape(-1)], t)

    def maskedFill(self, t, mask, value):
        x = self._g(t)
        return self._like(np.where(np.broadcast_to(self._g(mask).astype(bool), x.shape), value, x), t)

    def nonzero(self, t):  # flat indices + count, like the kernel
        flat = np.flatnonzero(self._g(t))
        return _H(count=int(flat.size), indices=self._reg(_F(flat), "float32"))

    # ── shape ─────────────────────────────────────────────────────────────
    def reshape(self, t, shape):
        return self._like(self._g(t).reshape([int(s) for s in shape]), t)

    def flatten(self, t, start=0, end=-1):
        x = self._g(t)
        if x.ndim == 0:
            return self._like(x.reshape(1), t)
        s, e = _ax(start, x.ndim), _ax(end, x.ndim)
        shp = list(x.shape[:s]) + [int(np.prod(x.shape[s:e + 1]))] + list(x.shape[e + 1:])
        return self._like(x.reshape(shp), t)

    def squeeze(self, t, dim=None):
        x = self._g(t)
        if dim is None:
            shp = [s for s in x.shape if s != 1]
        else:
            d = _ax(dim, x.ndim)
            shp = [s for i, s in enumerate(x.shape) if not (i == d and s == 1)]
        return self._like(x.reshape(shp or [1]), t)

    def unsqueeze(self, t, dim):
        x = self._g(t)
        return self._like(np.expand_dims(x, _ax(dim, x.ndim + 1)), t)

    def transpose2d(self, t): return self._like(self._g(t).T, t)

    def transpose(self, t, d0, d1):
        x = self._g(t)
        return self._like(np.swapaxes(x, _ax(d0, x.ndim), _ax(d1, x.ndim)), t)

    def permute(self, t, dims): return self._like(np.transpose(self._g(t), [int(d) for d in dims]), t)

    def select(self, t, dim, index):
        x = self._g(t)
        return self._like(np.take(x, int(index), axis=_ax(dim, x.ndim)), t)

    @staticmethod
    def _norm_slice(start, end, size, step):
        def ns(v, default_pos, default_neg):
            if v is None:
                return default_pos if step > 0 else default_neg
            v = v + size if v < 0 else v
            return max(0, min(size, v)) if step > 0 else max(-1, min(size - 1, v))
        return ns(start, 0, size - 1), ns(end, size, -1)

    def slice(self, t, dim, start=None, end=None, step=1):
        x = self._g(t)
        d = _ax(dim, x.ndim)
        s, e = self._norm_slice(start, end, x.shape[d], step)
        n = math.ceil(abs(e - s) / abs(step)) if (e - s) * step > 0 else 0
        idx = [slice(None)] * x.ndim
        idx[d] = np.arange(n) * step + s
        return self._like(x[tuple(idx)], t)

    def sliceBackward(self, gout, in_shape, sl_shape, dim, start, step):
        g = self._g(gout)
        out = np.zeros([int(s) for s in in_shape], g.dtype)
        d = _ax(dim, len(in_shape))
        idx = [slice(None)] * len(in_shape)
        idx[d] = np.arange(int(sl_shape[d])) * int(step) + int(start)
        np.add.at(out, tuple(idx), g.reshape([int(s) for s in sl_shape]))
        return self._reg(out, self._d(gout))

    def cat(self, ids, dim):
        ids = list(ids.to_py() if hasattr(ids, "to_py") else ids)
        xs = [self._g(i) for i in ids]
        return self._reg(np.concatenate(xs, axis=_ax(dim, xs[0].ndim)), self._d(ids[0]))

    def stack(self, ids, dim):
        ids = list(ids.to_py() if hasattr(ids, "to_py") else ids)
        xs = [self._g(i) for i in ids]
        return self._reg(np.stack(xs, axis=_ax(dim, xs[0].ndim + 1)), self._d(ids[0]))

    def expand(self, t, shape):
        x = self._g(t)
        shp = [int(s) for s in shape]
        lead = len(shp) - x.ndim
        shp = [x.shape[i - lead] if s == -1 else s for i, s in enumerate(shp)]
        return self._like(np.broadcast_to(x, shp), t)

    def repeat(self, t, sizes):
        x = self._g(t)
        sizes = [int(s) for s in sizes]
        if len(sizes) > x.ndim:
            x = x.reshape([1] * (len(sizes) - x.ndim) + list(x.shape))
        return self._like(np.tile(x, sizes), t)

    def indexSelect(self, t, dim, ind):
        x = self._g(t)
        return self._like(np.take(x, self._g(ind).reshape(-1).astype(np.int64), axis=_ax(dim, x.ndim)), t)

    def gather(self, t, dim, ind):
        x = self._g(t)
        return self._like(np.take_along_axis(x, self._g(ind).astype(np.int64), axis=_ax(dim, x.ndim)), t)

    def scatter(self, t, _dim, ind, src):  # flat-index scatter, like the kernel
        out = self._g(t).copy().reshape(-1)
        i = self._g(ind).reshape(-1).astype(np.int64)
        s = self._g(src).reshape(-1)
        out[i] = s[: len(i)]
        return self._reg(out.reshape(self._g(t).shape), self._d(t))

    def scatterAdd(self, t, _dim, ind, src):
        out = self._g(t).copy().reshape(-1).astype(np.float32 if _is_float(self._d(t)) else self._g(t).dtype)
        i = self._g(ind).reshape(-1).astype(np.int64)
        np.add.at(out, i, self._g(src).reshape(-1)[: len(i)])
        return self._reg(out.reshape(self._g(t).shape), self._d(t))

    def sort(self, t, dim):
        x = self._g(t)
        ax = _ax(dim, x.ndim)
        order = np.argsort(x, axis=ax, kind="stable")
        return [self._like(np.take_along_axis(x, order, axis=ax), t), self._reg(order, "int64")]

    def sortBackward(self, gout, ind, in_shape, dim):
        g = self._g(gout)
        i = self._g(ind).astype(np.int64)
        out = np.zeros([int(s) for s in in_shape], g.dtype)
        np.put_along_axis(out, i, g, axis=_ax(dim, len(in_shape)))
        return self._reg(out, self._d(gout))

    def topkBackward(self, gout, ind, in_shape, dim, k):
        g = self._g(gout)
        i = self._g(ind).astype(np.int64)
        out = np.zeros([int(s) for s in in_shape], g.dtype)
        np.put_along_axis(out, i, g, axis=_ax(dim, len(in_shape)))
        return self._reg(out, self._d(gout))

    def tril(self, t, diagonal=0): return self._like(np.tril(self._g(t), int(diagonal)), t)
    def triu(self, t, diagonal=0): return self._like(np.triu(self._g(t), int(diagonal)), t)

    def flip(self, t, dims):
        x = self._g(t)
        return self._like(np.flip(x, [_ax(int(d), x.ndim) for d in dims]) if x.ndim else x, t)

    def roll(self, t, shift):  # flat roll, like the kernel
        x = self._g(t)
        return self._like(np.roll(x.reshape(-1), int(shift)).reshape(x.shape), t)

    def _pad(self, t, l, r, top, bot, mode, value=0):
        x = self._g(t)
        pw = [(0, 0)] * (x.ndim - 2) + [(int(top), int(bot)), (int(l), int(r))]
        if mode == "constant":
            return self._like(np.pad(x, pw, mode="constant", constant_values=value), t)
        return self._like(np.pad(x, pw, mode=mode), t)

    def replicationPad(self, t, l, r, top, bot): return self._pad(t, l, r, top, bot, "edge")
    def reflectionPad(self, t, l, r, top, bot): return self._pad(t, l, r, top, bot, "reflect")
    def circularPad(self, t, l, r, top, bot): return self._pad(t, l, r, top, bot, "wrap")
    def constantPad(self, t, l, r, top, bot, value): return self._pad(t, l, r, top, bot, "constant", value)

    def upsample2d(self, t, oh, ow, mode, align_corners):
        x = _F(self._g(t))
        H, W = x.shape[-2:]
        oh, ow = int(oh), int(ow)
        if int(mode) == 0:  # nearest
            ri = np.minimum((np.arange(oh) * H / oh).astype(np.int64), H - 1)
            ci = np.minimum((np.arange(ow) * W / ow).astype(np.int64), W - 1)
            return self._like(x[..., ri[:, None], ci[None, :]], t)

        def coords(o, n):
            if align_corners:
                return np.arange(o) * ((n - 1) / (o - 1) if o > 1 else 0)
            return np.clip((np.arange(o) + 0.5) * n / o - 0.5, 0, n - 1)
        ys, xs = coords(oh, H), coords(ow, W)
        y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
        y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
        wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
        a = x[..., y0[:, None], x0[None, :]]
        b = x[..., y0[:, None], x1[None, :]]
        c = x[..., y1[:, None], x0[None, :]]
        d = x[..., y1[:, None], x1[None, :]]
        return self._like(a * (1 - wy) * (1 - wx) + b * (1 - wy) * wx + c * wy * (1 - wx) + d * wy * wx, t)

    # ── conv / pool / norm / embedding ────────────────────────────────────
    @staticmethod
    def _windows(x, kh, kw, sh, sw, ph, pw, dh=1, dw=1, pad_value=0.0):
        """(N, C, H, W) -> (N, C, OH, OW, kh, kw) view of the padded input."""
        xp = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)], constant_values=pad_value)
        N, C, H, W = xp.shape
        OH = (H - dh * (kh - 1) - 1) // sh + 1
        OW = (W - dw * (kw - 1) - 1) // sw + 1
        s = xp.strides
        return np.lib.stride_tricks.as_strided(
            xp, (N, C, OH, OW, kh, kw), (s[0], s[1], s[2] * sh, s[3] * sw, s[2] * dh, s[3] * dw), writeable=False)

    def conv2d(self, inp, w, bias, stride, padding, dilation, groups):
        x = _F(self._g(inp))
        K = _F(self._g(w))
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None]
        sh, sw = (list(stride) * 2)[:2]
        ph, pw = (list(padding) * 2)[:2]
        dh, dw = (list(dilation) * 2)[:2]
        G = int(groups)
        OC, ICg, kh, kw = K.shape
        win = self._windows(x, kh, kw, int(sh), int(sw), int(ph), int(pw), int(dh), int(dw))
        outs = []
        for g in range(G):
            wg = win[:, g * ICg:(g + 1) * ICg]
            kg = K[g * (OC // G):(g + 1) * (OC // G)]
            outs.append(np.einsum("ncpqij,ocij->nopq", wg, kg, optimize=True))
        y = np.concatenate(outs, axis=1)
        if bias is not None:
            b = np.asarray(list(bias.to_py() if hasattr(bias, "to_py") else bias), np.float32)
            y = y + b[None, :, None, None]
        return self._reg(y[0] if squeeze else y, self._d(inp))

    def conv2dInputBackward(self, gout, w, in_shape, gout_shape, stride, padding):
        g = _F(self._g(gout)).reshape([int(s) for s in gout_shape])
        K = _F(self._g(w))
        N, C, H, W = [int(s) for s in in_shape]
        sh, sw = (list(stride) * 2)[:2]
        ph, pw = (list(padding) * 2)[:2]
        OC, _, kh, kw = K.shape
        gx = np.zeros((N, C, H + 2 * ph, W + 2 * pw), np.float32)
        OH, OW = g.shape[2], g.shape[3]
        for i in range(kh):
            for j in range(kw):
                gx[:, :, i:i + sh * OH:sh, j:j + sw * OW:sw] += np.einsum("nopq,oc->ncpq", g, K[:, :, i, j])
        return self._reg(gx[:, :, ph:ph + H, pw:pw + W], self._d(gout))

    def conv2dWeightBackward(self, gout, inp, w_shape, gout_shape, in_shape, stride, padding):
        g = _F(self._g(gout)).reshape([int(s) for s in gout_shape])
        x = _F(self._g(inp)).reshape([int(s) for s in in_shape])
        OC, C, kh, kw = [int(s) for s in w_shape]
        sh, sw = (list(stride) * 2)[:2]
        ph, pw = (list(padding) * 2)[:2]
        win = self._windows(x, kh, kw, int(sh), int(sw), int(ph), int(pw))
        return self._reg(np.einsum("ncpqij,nopq->ocij", win, g, optimize=True), self._d(gout))

    def conv2dBiasBackward(self, gout, out_ch, gout_shape):
        g = _F(self._g(gout)).reshape([int(s) for s in gout_shape])
        return self._reg(g.sum(axis=(0, 2, 3)), self._d(gout))

    def _pool_args(self, inp, k, s, p):
        x = _F(self._g(inp))
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None]
        kh, kw = (list(k) * 2)[:2]
        sh, sw = (list(s) * 2)[:2]
        ph, pw = (list(p) * 2)[:2]
        return x, squeeze, int(kh), int(kw), int(sh), int(sw), int(ph), int(pw)

    def maxPool2d(self, inp, k, s, p, dil):
        x, sq, kh, kw, sh, sw, ph, pw = self._pool_args(inp, k, s, p)
        dh, dw = (list(dil) * 2)[:2]
        y = self._windows(x, kh, kw, sh, sw, ph, pw, int(dh), int(dw), -np.inf).max(axis=(4, 5))
        return self._reg(y[0] if sq else y, self._d(inp))

    def avgPool2d(self, inp, k, s, p, count_include_pad):
        x, sq, kh, kw, sh, sw, ph, pw = self._pool_args(inp, k, s, p)
        tot = self._windows(x, kh, kw, sh, sw, ph, pw).sum(axis=(4, 5))
        if count_include_pad or (ph == 0 and pw == 0):
            y = tot / (kh * kw)
        else:
            cnt = self._windows(np.ones_like(x[:1, :1]), kh, kw, sh, sw, ph, pw).sum(axis=(4, 5))
            y = tot / cnt
        return self._reg(y[0] if sq else y, self._d(inp))

    def pool2dBackward(self, gout, inp, k, s, p, mode, count_include_pad=True):
        """ai2ai: gradient of max/avg pool2d (not in the JS runtime)."""
        x, sq, kh, kw, sh, sw, ph, pw = self._pool_args(inp, k, s, p)
        g = _F(self._g(gout))
        if sq:
            g = g[None]
        N, C, H, W = x.shape
        OH, OW = g.shape[2], g.shape[3]
        gx = np.zeros((N, C, H + 2 * ph, W + 2 * pw), np.float32)
        if mode == "max":
            win = self._windows(x, kh, kw, sh, sw, ph, pw, pad_value=-np.inf).reshape(N, C, OH, OW, kh * kw)
            am = win.argmax(axis=-1)
            di, dj = am // kw, am % kw
            n, c, oh, ow = np.indices((N, C, OH, OW))
            np.add.at(gx, (n, c, oh * sh + di, ow * sw + dj), g)
        else:
            if count_include_pad or (ph == 0 and pw == 0):
                gg = g / (kh * kw)
            else:
                cnt = self._windows(np.ones((1, 1, H, W), np.float32), kh, kw, sh, sw, ph, pw).sum(axis=(4, 5))
                gg = g / cnt
            for i in range(kh):
                for j in range(kw):
                    gx[:, :, i:i + sh * OH:sh, j:j + sw * OW:sw] += gg
        gx = gx[:, :, ph:ph + H, pw:pw + W]
        return self._reg(gx[0] if sq else gx, self._d(inp))

    def batchNorm(self, inp, w, b, rm, rv, eps):
        x = _F(self._g(inp))
        C = x.shape[1]
        shp = [1, C] + [1] * (x.ndim - 2)
        get = lambda tid, d: _F(self._g(tid)).reshape(shp) if tid is not None else np.full(shp, d, np.float32)
        y = (x - get(rm, 0)) / np.sqrt(get(rv, 1) + np.float32(eps)) * get(w, 1) + get(b, 0)
        return self._reg(y, self._d(inp))

    def layerNorm(self, inp, norm_shape, gamma, beta, eps):
        x = _F(self._g(inp))
        n = int(np.prod([int(s) for s in norm_shape]))
        xr = x.reshape(-1, n)
        mu = xr.mean(axis=1, keepdims=True)
        var = ((xr - mu) ** 2).mean(axis=1, keepdims=True)
        y = (xr - mu) / np.sqrt(var + np.float32(eps))
        if gamma is not None:
            y = y * _F(self._g(gamma)).reshape(1, n)
        if beta is not None:
            y = y + _F(self._g(beta)).reshape(1, n)
        return self._reg(y.reshape(x.shape), self._d(inp))

    def embedding(self, w, ind, num, dim, padding_idx):
        W = self._g(w)
        i = self._g(ind).astype(np.int64)
        out = W[np.clip(i, 0, int(num) - 1)]  # PyTorch: the padding row is returned as stored
        return self._reg(out, self._d(w))

    # ── linalg ────────────────────────────────────────────────────────────
    def cholesky(self, t): return self._like(np.linalg.cholesky(self._g(t).astype(np.float64)), t)

    def triangularSolve(self, a, b, upper):
        A, B = self._g(a).astype(np.float64), self._g(b).astype(np.float64)
        A = np.triu(A) if upper else np.tril(A)
        return self._like(np.linalg.solve(A, B), b)

    def gramSchmidt(self, t):
        q, _ = np.linalg.qr(self._g(t).astype(np.float64))
        return self._like(q, t)

    def lu(self, t):
        raise NotImplementedError("lu (cpu backend)")

    def jacobiRotate(self, *a):
        raise NotImplementedError("jacobiRotate (cpu backend)")


_INSTANCE: NumpyRuntime | None = None


def get_numpy_runtime() -> NumpyRuntime:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = NumpyRuntime()
    return _INSTANCE
