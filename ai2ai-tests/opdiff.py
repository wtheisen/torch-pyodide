# Per-op differential test: WebGPU runtime vs numpy runtime, same inputs.
import math
from js import globalThis
from pyodide.ffi import run_sync, to_js
from torch._numpy_runtime import NumpyRuntime

JS = getattr(globalThis, "__torch_pyodide_runtime__")
NP = NumpyRuntime()


def py(v):
    return v.to_py() if hasattr(v, "to_py") else v


def jcall(name, *args):
    a = [to_js(x) if isinstance(x, (list, tuple)) else x for x in args]
    r = getattr(JS, name)(*a)
    return py(run_sync(r)) if hasattr(r, "then") else py(r)


def hget(h, k):
    try:
        return h[k]
    except Exception:
        return getattr(h, k)


def mk(data, shape, dtype="float32"):
    j = jcall("tensorFromData", data, shape, dtype)
    j = {"id": hget(j, "id")}
    n = NP.tensorFromData(data, shape, dtype)
    return j["id"], n["id"]


def vals(n, k=1.0, s=0):
    return [round(math.sin(1.7 * i + s) * k, 4) for i in range(n)]


def readj(h):
    return list(py(run_sync(JS.toList(hget(h, "id"))))), list(py(hget(h, "shape"))), str(hget(h, "dtype"))


def readn(h):
    return NP.toList(h["id"]), list(h["shape"]), str(h["dtype"])


bad, ok = [], 0


def cmp(name, rj, rn, tol=2e-4):
    global ok
    (vj, sj, dj), (vn, sn, dn) = rj, rn
    issues = []
    if sj != sn:
        issues.append(f"shape {sj} vs {sn}")
    if dj != dn:
        issues.append(f"dtype {dj} vs {dn}")
    if len(vj) != len(vn):
        issues.append(f"len {len(vj)} vs {len(vn)}")
    else:
        for a, b in zip(vj, vn):
            a, b = float(a), float(b)
            if (math.isnan(a) and math.isnan(b)) or a == b:
                continue
            if abs(a - b) > tol * max(1.0, abs(a), abs(b)):
                issues.append(f"vals {[round(float(x),4) for x in vj[:6]]} vs {[round(float(x),4) for x in vn[:6]]}")
                break
    if issues:
        bad.append((name, "; ".join(issues)))
    else:
        ok += 1


def run(name, args_fn, extra=()):
    """args_fn(side) -> positional args (side 0 = js ids, 1 = numpy ids)."""
    try:
        hj = jcall(name, *args_fn(0), *extra)
    except Exception as e:
        hj = ("ERR", str(e)[:80])
    try:
        hn = getattr(NP, name)(*args_fn(1), *extra)
    except Exception as e:
        hn = ("ERR", str(e)[:80])
    if isinstance(hj, tuple) or isinstance(hn, tuple):
        if isinstance(hj, tuple) and isinstance(hn, tuple):
            ok_ = True
        else:
            bad.append((name, f"js={hj if isinstance(hj, tuple) else 'ok'} np={hn if isinstance(hn, tuple) else 'ok'}"))
        return
    if isinstance(hj, list):
        for i, (a, b) in enumerate(zip(hj, hn)):
            cmp(f"{name}[{i}]", readj(a), readn(b))
        return
    if hj is None:
        return
    try:
        cmp(name, readj(hj), readn(hn))
    except Exception as e:
        bad.append((name, "CMP ERR " + str(e)[:80]))


A = mk(vals(24, 2, 0), [2, 3, 4])
B = mk(vals(24, 2, 1), [2, 3, 4])
Bpos = mk([abs(v) + 0.5 for v in vals(24, 2, 1)], [2, 3, 4])
Apos = mk([abs(v) + 0.1 for v in vals(24, 0.9, 0)], [2, 3, 4])
Brow = mk(vals(4, 1, 2), [4])
M1 = mk(vals(12, 1, 3), [3, 4])
M2 = mk(vals(20, 1, 4), [4, 5])
I = mk([2, 0, 1, 3, 1, 0], [6], "int32")
L = mk(vals(15, 2, 5), [5, 3])
T = mk([2, 0, 1, 1, 2], [5], "int32")
MASK = mk([1, 0, 0, 1] * 6, [2, 3, 4], "bool")
IMG = mk(vals(2 * 2 * 6 * 6, 1, 6), [2, 2, 6, 6])
KER = mk(vals(3 * 2 * 3 * 3, 0.5, 7), [3, 2, 3, 3])
W = mk(vals(40, 1, 8), [10, 4])
IDX = mk([1, 3, 9, 0, 3], [5], "int32")
SQ = mk([4, 2, 0.6, 2, 5, 1.2, 0.6, 1.2, 3], [3, 3])

ids = lambda t: (lambda side: [t[side]])
two = lambda a, b: (lambda side: [a[side], b[side]])

for op in ["add", "sub", "mul", "div", "maximum", "minimum", "atan2", "hypot", "logaddexp", "fmod",
           "remainder", "copysign", "eq", "ne", "lt", "le", "gt", "ge", "trueDivide", "floorDivide"]:
    run(op, two(A, B))
    run(op + "~bcast", two(A, mk(vals(4, 1, 2), [4]))) if False else None
for op in ["add", "mul", "sub", "div", "gt"]:
    run(op, two(A, Brow))
run("pow", two(Apos, B))
run("xlogy", two(Apos, Bpos))
run("logicalAnd", two(MASK, MASK))
run("where", lambda s: [MASK[s], A[s], B[s]])
run("matmul", two(M1, M2))
run("matmul", two(mk(vals(24, 1, 9), [2, 3, 4]), mk(vals(40, 1, 10), [2, 4, 5])))
run("mulScalar", ids(A), (2.5,))
run("clamp", ids(A), (-0.5, 0.7))
run("lerpScalar", two(A, B), (0.3,))
run("addcmul", lambda s: [A[s], B[s], Bpos[s]], (0.5,))
run("addcdiv", lambda s: [A[s], B[s], Bpos[s]], (0.5,))

for op in ["relu", "abs", "exp", "neg", "sigmoid", "tanh", "sin", "cos", "gelu", "silu", "floor", "ceil",
           "round", "square", "sign", "softplus", "mish", "hardsigmoid", "hardswish", "softsign", "tanhshrink",
           "erf", "erfc", "atan", "sinh", "trunc", "frac", "expm1", "deg2rad", "isnan", "isfinite", "logicalNot", "asinh"]:
    run(op, ids(A))
for op in ["sqrt", "log", "rsqrt", "reciprocal", "log2", "log10", "log1p", "lgamma", "digamma", "acosh"]:
    run(op, ids(Bpos))
run("leakyRelu", ids(A), (0.1,))
run("fill", ids(A), (3.0,))
for op in ["sum", "mean", "prod", "min", "max", "argmax", "argmin", "any", "all", "cumsum", "cumprod"]:
    run(op, ids(Apos if op in ("prod", "cumprod") else A))
for d in [0, 1, 2, -1]:
    run("sumDim", ids(A), (d, False))
    run("meanDim", ids(A), (d, True))
    run("softmax", ids(A), (d,))
    run("logSoftmax", ids(A), (d,))
LS = mk(vals(15, 1, 11), [5, 3])
run("nllLoss", two(LS, T))
run("nllLossReduced", two(LS, T), ("mean",))
run("crossEntropy", two(L, T))
G1 = mk([1.0], [1])
G5 = mk(vals(5, 1, 12), [5])
for red in ["mean", "sum"]:
    run("crossEntropyBackward", lambda s: [G1[s], L[s], T[s]], (red,))
run("crossEntropyBackward", lambda s: [G5[s], L[s], T[s]], ("none",))
SM = mk([0.2, 0.3, 0.5] * 5, [5, 3])
GG = mk(vals(15, 1, 13), [5, 3])
run("softmaxBackward", two(GG, SM), (5, 3))
run("logSoftmaxBackward", two(GG, SM), (5, 3))
run("nllLossBackward", ids(T), (5, 3, 0.2))
run("maxMinBackward", lambda s: [A[s], G1[s]], ("max",))
run("maxMinBackward", lambda s: [A[s], G1[s]], ("min",))
run("maskedFill", two(A, MASK), (-9.0,))
run("maskedSelect", two(A, MASK))
run("reshape", ids(A), ([4, 6],))
run("flatten", ids(A), (1, 2))
run("squeeze", ids(mk(vals(6), [1, 6, 1])), ())
run("squeeze", ids(mk(vals(6), [1, 6, 1])), (2,))
run("unsqueeze", ids(A), (1,))
run("transpose", ids(A), (0, 2))
run("transpose2d", ids(M1))
run("permute", ids(A), ([2, 0, 1],))
run("select", ids(A), (1, 2))
for args in [(2, 1, 3, 1), (1, 0, 3, 2), (2, -3, None, 1), (0, 0, 2, 1)]:
    run("slice", ids(A), args)
run("cat", lambda s: [[A[s], B[s]]], (1,))
run("stack", lambda s: [[A[s], B[s]]], (0,))
run("expand", ids(mk(vals(4), [1, 4])), ([3, 4],))
run("repeat", ids(M1), ([2, 1],))
run("indexSelect", two(W, IDX), ()) if False else run("indexSelect", lambda s: [W[s], 0, IDX[s]])
GI = mk([0, 2, 1, 1, 0, 2, 2, 1, 0, 1, 1, 0], [3, 4], "int32")
run("gather", lambda s: [M1[s], 0, GI[s]])
run("scatter", lambda s: [mk(vals(10), [10])[s], 0, IDX[s], mk(vals(5, 3), [5])[s]])
run("scatterAdd", lambda s: [mk([0.0] * 10, [10])[s], 0, mk([1, 1, 3, 0, 1], [5], "int32")[s], mk(vals(5, 3), [5])[s]])
run("sort", ids(A), (2,))
run("sort", ids(A), (0,))
run("tril", ids(M1), (0,))
run("triu", ids(M1), (1,))
run("flip", ids(A), ([0, 2],))
run("roll", ids(A), (3,))
for p in ["replicationPad", "reflectionPad", "circularPad"]:
    run(p, ids(IMG), (1, 2, 1, 0))
run("constantPad", ids(IMG), (1, 1, 2, 0, 0.5))
run("upsample2d", ids(IMG), (12, 9, 0, 0))
run("upsample2d", ids(IMG), (12, 9, 1, 0))
run("upsample2d", ids(IMG), (12, 9, 1, 1))
run("conv2d", lambda s: [IMG[s], KER[s]], ([0.1, -0.2, 0.3], [1, 1], [1, 1], [1, 1], 1))
run("conv2d", lambda s: [IMG[s], KER[s]], (None, [2, 2], [0, 0], [1, 1], 1))
GO = mk(vals(2 * 3 * 6 * 6, 1, 14), [2, 3, 6, 6])
run("conv2dInputBackward", lambda s: [GO[s], KER[s]], ([2, 2, 6, 6], [2, 3, 6, 6], [1, 1], [1, 1]))
run("conv2dWeightBackward", lambda s: [GO[s], IMG[s]], ([3, 2, 3, 3], [2, 3, 6, 6], [2, 2, 6, 6], [1, 1], [1, 1]))
run("conv2dBiasBackward", ids(GO), (3, [2, 3, 6, 6]))
run("maxPool2d", ids(IMG), ([2, 2], [2, 2], [0, 0], [1, 1]))
run("maxPool2d", ids(IMG), ([3, 3], [2, 2], [1, 1], [1, 1]))
run("avgPool2d", ids(IMG), ([2, 2], [2, 2], [0, 0], True))
run("avgPool2d", ids(IMG), ([3, 3], [2, 2], [1, 1], False))
GAM = mk(vals(4, 1, 15), [4])
BET = mk(vals(4, 1, 16), [4])
run("layerNorm", lambda s: [A[s], [4], GAM[s], BET[s]], (1e-5,))
CH = lambda k: mk(vals(2, 1, k), [2])
RV = mk([0.5, 2.0], [2])
run("batchNorm", lambda s: [IMG[s], CH(17)[s], CH(18)[s], CH(19)[s], RV[s]], (1e-5,))
run("embedding", lambda s: [W[s], IDX[s]], (10, 4, 0xFFFFFFFF))
run("embedding", lambda s: [W[s], IDX[s]], (10, 4, 3))
run("cholesky", ids(SQ))
run("gramSchmidt", ids(SQ))
run("pdist", ids(M1), (2.0,))
run("nonzero", ids(MASK))


# in-place optimizer steps
def step(name, nstate, hp):
    P = mk(vals(12, 1, 20), [12]); G = mk(vals(12, 0.5, 21), [12])
    S = [mk([abs(v) for v in vals(12, 0.1, 22 + k)], [12]) for k in range(nstate)]
    try:
        jcall(name, P[0], G[0], *[s[0] for s in S], *hp)
        getattr(NP, name)(P[1], G[1], *[s[1] for s in S], *hp)
        for k, t in enumerate([P] + S):
            cmp(f"{name}#{k}", (list(py(run_sync(JS.toList(t[0])))), [12], "float32"), (NP.toList(t[1]), [12], "float32"))
    except Exception as e:
        bad.append((name, "ERR " + str(e)[:100]))


step("sgdStep", 1, (0.1, 0.9, 0.01, 0.0, False))
step("sgdStep", 1, (0.1, 0.9, 0.0, 0.1, True))
step("adamStep", 2, (0.01, 0.9, 0.999, 1e-8, 0.01, 0.05, 1.3))
step("adamWStep", 2, (0.01, 0.9, 0.999, 1e-8, 0.01, 0.05, 1.3))
step("rmspropStep", 2, (0.01, 0.99, 1e-8, 0.01, 0.9))
step("adagradStep", 1, (0.01, 1e-10, 0.01))
step("adamaxStep", 2, (0.01, 0.9, 0.999, 1e-8, 0.01, 0.05, 1.2))
step("nadamStep", 2, (0.01, 0.9, 0.999, 1e-8, 0.01, 0.05, 0.9))
step("radamStep", 2, (0.01, 0.9, 0.999, 1e-8, 0.01, 0.05, 0.5, 0.6))

print("ok", ok, "bad", len(bad))
for b in bad:
    print("BAD", b[0], "|", b[1])
