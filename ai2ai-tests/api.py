# Broad API comparison: run under real PyTorch and under the fork; outputs must match.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def det(shape, k=1.0, s=0):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor([math.sin(1.3 * i + 0.7 * s + 0.1) * k for i in range(n)]).reshape(*shape)


def r(x):
    if isinstance(x, torch.Tensor):
        v = x.detach().reshape(-1).tolist()
        return [list(x.shape)] + [round(float(t), 4) for t in v[:6]] + [round(sum(abs(float(t)) for t in v), 3)]
    if isinstance(x, (list, tuple)):
        return [r(y) for y in x]
    return round(float(x), 4) if isinstance(x, float) else x


def T(name, f):
    try:
        print(name, r(f()))
    except Exception as e:
        print(name, "FAIL", type(e).__name__, str(e)[:90])


A = det((2, 3, 4), 2.0, 1)
B = det((2, 3, 4), 2.0, 2)
P = det((2, 3, 4), 1.0, 3).abs() + 0.2
M = det((3, 4), 1.0, 4)

# elementwise + broadcasting
T("add_bcast", lambda: A + det((4,), 1, 5))
T("sub_scalar", lambda: 3.0 - A)
T("rdiv", lambda: 1.0 / P)
T("pow_scalar", lambda: P ** 2.5)
T("rpow", lambda: 2.0 ** M)
T("neg_abs", lambda: (-A).abs())
T("clamp", lambda: A.clamp(-0.5, 1.0))
T("clamp_min", lambda: A.clamp(min=0.0))
T("where", lambda: torch.where(A > 0, A, B))
T("exp_log", lambda: torch.log(torch.exp(M)))
T("sqrt", lambda: P.sqrt())
T("floordiv", lambda: torch.floor(A / 0.7))
T("comparisons", lambda: [(A > B).float(), (A <= 0.3).float(), (A == A).float()])
T("logical", lambda: ((A > 0) & (B > 0)).float())
T("lerp", lambda: torch.lerp(A, B, 0.25))
# reductions
for d in [0, 1, -1]:
    T(f"sum_dim{d}", lambda d=d: A.sum(dim=d))
    T(f"mean_keep{d}", lambda d=d: A.mean(dim=d, keepdim=True))
    T(f"max_dim{d}", lambda d=d: A.max(dim=d))
    T(f"argmax_dim{d}", lambda d=d: A.argmax(dim=d))
    T(f"softmax{d}", lambda d=d: torch.softmax(A, dim=d))
    T(f"logsoftmax{d}", lambda d=d: F.log_softmax(A, dim=d))
    T(f"cumsum{d}", lambda d=d: A.cumsum(dim=d))
T("sum_all", lambda: A.sum())
T("mean_all", lambda: A.mean())
T("std_var", lambda: [A.std(), A.var(), A.var(dim=1)])
T("norm", lambda: [A.norm(), M.norm(dim=1)])
T("argmax_all", lambda: A.argmax())
T("min_max_all", lambda: [A.min(), A.max()])
T("topk", lambda: torch.topk(M, 2, dim=1))
T("sort", lambda: torch.sort(M, dim=1, descending=True))
T("any_all", lambda: [(A > 1.5).any(), (A > -3).all()])
# shape/view/indexing
T("view_reshape", lambda: [A.view(6, 4), A.reshape(-1), A.flatten(1)])
T("permute", lambda: A.permute(2, 0, 1))
T("transpose", lambda: A.transpose(1, 2))
T("unsqueeze_squeeze", lambda: [A.unsqueeze(0).shape, A[:, :1].squeeze(1)])
T("index_int", lambda: [A[1], A[0, 2], A[:, 1, 3]])
T("slice", lambda: [A[:, 1:], A[..., ::2], A[:, -1]])
T("bool_index", lambda: A[A > 1.0])
T("tensor_index", lambda: M[torch.tensor([2, 0])])
T("cat3", lambda: torch.cat([A, B, A], dim=1))
T("stack3", lambda: torch.stack([M, M * 2, M * 3], dim=0))
T("split_chunk", lambda: [torch.split(M, 2, dim=1), torch.chunk(A, 3, dim=1)])
T("gather", lambda: torch.gather(M, 1, torch.tensor([[0, 3], [1, 1], [2, 0]])))
T("expand_repeat", lambda: [det((1, 4)).expand(3, 4), det((2, 1)).repeat(1, 3)])
T("flip_roll", lambda: [torch.flip(M, [1]), torch.roll(M, 1, dims=1)])
T("tril_triu", lambda: [torch.tril(M), torch.triu(M, 1)])
T("one_hot", lambda: F.one_hot(torch.tensor([0, 3, 1]), 4))
T("arange_linspace", lambda: [torch.arange(0, 5, 2), torch.linspace(0, 1, 5)])
T("zeros_ones_full", lambda: [torch.zeros(2, 3), torch.ones_like(M), torch.full((2, 2), 7.0), torch.eye(3)])
T("matmul_variants", lambda: [M @ M.T, torch.mm(M, M.T), (A @ det((4, 2))), torch.bmm(A, A.transpose(1, 2))])
T("outer_dot", lambda: [torch.outer(det((3,)), det((2,))), torch.dot(det((4,)), det((4,), 1, 2))])
T("einsum", lambda: torch.einsum("bij,bkj->bik", A, B))
# losses
y_cls = torch.tensor([1, 0, 3])
logits = det((3, 4), 2.0, 6)
T("cross_entropy", lambda: [F.cross_entropy(logits, y_cls), F.cross_entropy(logits, y_cls, reduction="sum")])
T("nll", lambda: F.nll_loss(F.log_softmax(logits, dim=1), y_cls))
T("mse_l1", lambda: [F.mse_loss(A, B), F.l1_loss(A, B), F.mse_loss(A, B, reduction="sum")])
T("bce", lambda: [F.binary_cross_entropy(torch.sigmoid(A), (B > 0).float()), F.binary_cross_entropy_with_logits(A, (B > 0).float())])
T("smooth_l1_huber", lambda: [F.smooth_l1_loss(A, B), F.huber_loss(A, B)])
# activations
for name in ["relu", "sigmoid", "tanh", "gelu", "silu", "softplus", "elu", "leaky_relu"]:
    T(name, lambda name=name: getattr(F, name)(A))
# layers
torch.manual_seed(0)
x4 = det((2, 2, 6, 6), 1.0, 7)


def setw(m, s):
    m.load_state_dict({k: det(tuple(v.shape), 0.4, s + i) for i, (k, v) in enumerate(sorted(m.state_dict().items()))
                       if "running" not in k and "num_batches" not in k}, strict=False)
    return m


T("conv_stride_pad", lambda: setw(nn.Conv2d(2, 3, 3, stride=2, padding=1), 8)(x4))
T("conv_nobias", lambda: setw(nn.Conv2d(2, 4, 2, bias=False), 9)(x4))
T("conv_5x5", lambda: setw(nn.Conv2d(2, 1, 5, padding=2), 10)(x4))
T("pools", lambda: [F.max_pool2d(x4, 2), F.avg_pool2d(x4, 3, stride=1), F.max_pool2d(x4, 3, stride=2, padding=1)])
T("batchnorm_train_eval", lambda: (lambda bn: [bn(x4), bn.running_mean, bn.eval()(x4)])(setw(nn.BatchNorm2d(2), 11)))
T("layernorm", lambda: setw(nn.LayerNorm(4), 12)(A))
T("dropout_eval", lambda: nn.Dropout(0.5).eval()(A))
T("embedding_pad", lambda: setw(nn.Embedding(10, 3, padding_idx=0), 13)(torch.tensor([[0, 2, 9], [4, 0, 1]])))
T("upsample", lambda: [F.interpolate(x4, scale_factor=2, mode="nearest"), F.interpolate(x4, size=(9, 9), mode="bilinear", align_corners=False)])
T("flatten_linear", lambda: setw(nn.Sequential(nn.Flatten(), nn.Linear(72, 5)), 14)(x4))


# autograd through many ops
def grads():
    a = det((3, 4), 1.0, 15).requires_grad_()
    b = det((4, 2), 1.0, 16).requires_grad_()
    h = torch.tanh(a @ b)
    z = (h.softmax(dim=1) * torch.sigmoid(h)).sum() + (a ** 2).mean() + a.abs().max() + h[:, 1].sum()
    z = z + torch.cat([a, a * 2], dim=0).std() + F.cross_entropy(a, torch.tensor([0, 3, 1]))
    z.backward()
    return [z, a.grad, b.grad]


T("autograd_mix", grads)


def conv_grads():
    x = det((2, 2, 6, 6), 1.0, 17).requires_grad_()
    c = setw(nn.Conv2d(2, 3, 3, padding=1), 18)
    bn = setw(nn.BatchNorm2d(3), 19)
    y = F.max_pool2d(F.relu(bn(c(x))), 2).pow(2).sum()
    y.backward()
    return [y, x.grad, c.weight.grad, c.bias.grad, bn.weight.grad]


T("autograd_cnn_bn", conv_grads)


def train_adamw():
    m = setw(nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.LayerNorm(8), nn.Linear(8, 3)), 20)
    opt = torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.1)
    xs, ys = det((16, 4), 1, 21), torch.tensor([i % 3 for i in range(16)])
    for _ in range(30):
        opt.zero_grad()
        loss = F.cross_entropy(m(xs), ys)
        loss.backward()
        opt.step()
    return loss


T("train_adamw_gelu_ln", train_adamw)


def train_rmsprop_sched():
    m = setw(nn.Linear(4, 1), 22)
    opt = torch.optim.RMSprop(m.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    xs, ys = det((16, 4), 1, 23), det((16, 1), 1, 24)
    for _ in range(25):
        opt.zero_grad()
        loss = F.mse_loss(m(xs), ys)
        loss.backward()
        opt.step()
        sch.step()
    return [loss, opt.param_groups[0]["lr"]]


T("train_rmsprop_steplr", train_rmsprop_sched)


def lstm_seq():
    l = setw(nn.LSTM(3, 4, num_layers=2, batch_first=True), 25)
    x = det((2, 5, 3), 1, 26)
    o, (h, c) = l(x)
    o.sum().backward()
    return [o, h, c, l.weight_hh_l1.grad]


T("lstm_2layer", lstm_seq)


def gru_bidir():
    g = setw(nn.GRU(3, 2, batch_first=True, bidirectional=True), 27)
    o, h = g(det((1, 4, 3), 1, 28))
    return [o, h]


T("gru_bidir", gru_bidir)

with torch.no_grad():
    T("no_grad", lambda: (det((2, 2)).requires_grad_() * 2).requires_grad)
T("item_tolist", lambda: [det((1,)).item(), M[0].tolist()])
T("dtype_casts", lambda: [M.int(), (M > 0).long().sum(), M.double().dtype == torch.float64, torch.tensor([1, 2]).dtype])


# float64 stays float64, Python scalars are "weak", int tensor * float scalar is float
def f64():
    x = torch.tensor(1.36, dtype=torch.float64)
    m = M.double()
    return [x.item() == 1.36, str((x + 0.2).dtype), str((1 - m).dtype), str((m @ m.T).sum().dtype),
            (m.sum() * 1e-5).item(), torch.tensor([1, 2]) * 0.5, (torch.tensor([1, 2]) < 1.5).float()]


T("float64_weak_scalars", f64)


# item and slice assignment, recorded by autograd
def setitem():
    t = torch.zeros(3, 4)
    t[1, 2] = 5.0
    t[0] = torch.tensor([1., 2., 3., 4.])
    t[:, 0] = t[:, 0] / 100
    t[t > 3] = -1.0
    t[torch.tensor([2, 0]), 1] = 7.0
    K = det((2, 2), 1.0, 30).requires_grad_()
    img = det((4, 4), 1.0, 31)
    out = torch.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            out[i, j] = (img[i:i + 2, j:j + 2] * K).sum()
    (out ** 2).mean().backward()
    return [t, out, K.grad]


T("setitem_autograd", setitem)


# in-place arithmetic, 0-d scalars, integer sums, LAPACK-backed eigh/svd
def inplace():
    a = torch.tensor(2.)
    x = M.clone() if hasattr(M, "clone") else M * 1
    y = x
    x -= 1
    x.mul_(2)
    C = torch.zeros(3, 3)
    for i, j in [(0, 1), (1, 2), (0, 1)]:
        C[i, j] += 1
    w = det((4,), 1.0, 32).requires_grad_()
    xs, ys = det((5, 4), 1.0, 33), det((5,), 1.0, 34)
    for _ in range(3):
        loss = ((xs @ w - ys) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= 0.1 * w.grad
            w.grad.zero_()
    S = det((4, 3), 1.0, 35)
    return [list((a + 1).shape), y, C, (M > 0).sum(), w, torch.linalg.eigh(S.T @ S)[0], torch.linalg.svd(S)[1]]


T("inplace_linalg", inplace)
