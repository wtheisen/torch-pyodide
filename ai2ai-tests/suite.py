import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

R = {}
print("order", [n for n,_ in nn.Sequential(nn.Linear(2,2)).named_parameters()], [n for n,_ in nn.LSTM(2,2).named_parameters()])
def det(shape, k=1.0, s=0):
    n = 1
    for d in shape: n *= d
    vals = [math.sin(1.3 * i + 0.7 * s + 0.1) * k for i in range(n)]
    return torch.tensor(vals).reshape(*shape)

def setp(m, s=0, k=0.5):
    sd = {}
    for i, (name, p) in enumerate(sorted(m.named_parameters(), key=lambda t: t[0])):
        sd[name] = det(tuple(p.shape), k, s + i)
    m.load_state_dict(sd, strict=False)

def r(x, n=4):
    if isinstance(x, torch.Tensor): x = x.detach().reshape(-1).tolist()
    if isinstance(x, list): return [round(float(v), n) for v in x[:8]]
    return round(float(x), n)

def test(name):
    def deco(f):
        t = time.time()
        try:
            out = f()
            R[name] = out
            print(f"{name}: {out}  ({time.time()-t:.2f}s)")
        except Exception as e:
            R[name] = 'FAIL'
            print(f"{name}: FAIL {type(e).__name__}: {str(e)[:200]}")
        return f
    return deco

X = det((12, 6), 1.0, 1)
y = (det((12, 1), 1.0, 2) > 0).float()

@test("mlp_forward_backward")
def _():
    m = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid()); setp(m, 3)
    p = m(X); loss = nn.BCELoss()(p, y); loss.backward()
    return [r(loss), r(m[0].weight.grad)[:4], r(m[2].bias.grad)]

@test("mlp_sgd_50")
def _():
    m = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 1)); setp(m, 3)
    opt = torch.optim.SGD(m.parameters(), lr=0.5, momentum=0.9)
    for _ in range(50):
        opt.zero_grad(); loss = nn.BCEWithLogitsLoss()(m(X), y); loss.backward(); opt.step()
    return r(loss)

@test("mlp_adam_50")
def _():
    m = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 1)); setp(m, 5)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(50):
        opt.zero_grad(); loss = F.mse_loss(m(X), y); loss.backward(); opt.step()
    return r(loss)

@test("cnn_conv_pool_ce")
def _():
    img = det((3, 1, 8, 8), 1.0, 7)
    m = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32, 10)); setp(m, 9)
    out = m(img); loss = nn.CrossEntropyLoss()(out, torch.tensor([3, 1, 7])); loss.backward()
    return [r(loss), r(m[0].weight.grad)[:4], list(out.shape)]

@test("conv_stride_avgpool")
def _():
    img = det((1, 1, 8, 8), 1.0, 4); c = nn.Conv2d(1, 3, 3, stride=2); setp(c, 2)
    o = nn.AvgPool2d(2)(c(img)); return [list(o.shape), r(o)]

@test("embedding_rnn")
def _():
    emb = nn.Embedding(22, 2); rnn = nn.RNN(2, 2, batch_first=True); head = nn.Linear(2, 22)
    for i, mod in enumerate([emb, rnn, head]): setp(mod, 10 + i)
    ids = torch.tensor([[0, 0, 1, 2, 3, 4]]); out, h = rnn(emb(ids))
    loss = F.cross_entropy(head(out[0, :-1]), ids[0, 1:]); loss.backward()
    return [r(h), r(loss), r(rnn.weight_hh_l0.grad)]

@test("lstm_gru")
def _():
    x = det((1, 5, 3), 1.0, 20); l = nn.LSTM(3, 4, batch_first=True); g = nn.GRU(3, 4, batch_first=True)
    setp(l, 21); setp(g, 22)
    o1, (h1, c1) = l(x); o2, h2 = g(x); (o1.sum() + o2.sum()).backward()
    return [r(h1), r(c1)[:4], r(h2), r(l.weight_ih_l0.grad)[:4]]

@test("clip_grad_norm")
def _():
    m = nn.Linear(6, 1); setp(m, 30); (m(X * 50).pow(2).mean()).backward()
    n = nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    tot = math.sqrt(sum(float((p.grad ** 2).sum()) for p in m.parameters()))
    return [r(n, 2), r(tot)]

@test("mha_causal")
def _():
    x = det((1, 7, 4), 1.0, 40); mha = nn.MultiheadAttention(4, 2, batch_first=True); setp(mha, 41)
    mask = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
    o, w = mha(x, x, x, attn_mask=mask); o.sum().backward()
    return [r(o), r(w[0, 3]), r(mha.in_proj_weight.grad)[:4]]

@test("manual_attention_layernorm")
def _():
    x = det((7, 4), 1.0, 50); Wq, Wk, Wv = det((4, 4), .5, 51), det((4, 4), .5, 52), det((4, 4), .5, 53)
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    A = torch.softmax(Q @ K.T / math.sqrt(4), dim=-1); out = nn.LayerNorm(4)(A @ V + x)
    return [r(A[6]), r(out[0])]

@test("transformer_encoder_layer")
def _():
    x = det((1, 7, 8), 1.0, 60)
    t = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True, norm_first=True); setp(t, 61, 0.3)
    o = t(x, src_mask=nn.Transformer.generate_square_subsequent_mask(7)); return r(o)

@test("dataloader")
def _():
    ds = torch.utils.data.TensorDataset(X, y); dl = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=False)
    return [list(b[0].shape) for b in dl]

@test("module_api")
def _():
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.fc1 = nn.Linear(6, 3); self.fc2 = nn.Linear(3, 1)
        def forward(self, x): return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
    n = Net(); setp(n, 70); n.eval()
    with torch.no_grad(): p = n(X)
    return [sum(q.numel() for q in n.parameters()), r(p)[:4], sorted(n.state_dict().keys())]

@test("timing_train_141x6_300steps")
def _():
    Xb = det((141, 6), 1.0, 80); yb = (det((141, 1), 1.0, 81) > 0).float()
    m = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)); setp(m, 82, 0.2)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    for _ in range(300):
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(m(Xb), yb); loss.backward(); opt.step()
    return r(loss)

@test("defaults_seq_first")
def _():
    x = det((5, 2, 3), 1.0, 90); rnn = nn.RNN(3, 4); setp(rnn, 91)
    o, h = rnn(x); mha = nn.MultiheadAttention(4, 2); setp(mha, 92)
    a, w = mha(o, o, o); return [list(o.shape), r(h), list(a.shape), r(a), list(w.shape)]

@test("encoder_postnorm_and_stack")
def _():
    x = det((2, 7, 8), 1.0, 95)
    layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True); setp(layer, 96, 0.3)
    return r(layer(x, src_mask=nn.Transformer.generate_square_subsequent_mask(7)))
