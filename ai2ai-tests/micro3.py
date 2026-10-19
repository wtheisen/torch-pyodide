import torch, torch.nn as nn, torch.nn.functional as F, math
def det(shape, k=1.0, s=0):
    n=1
    for d in shape: n*=d
    return torch.tensor([math.sin(1.3*i+0.7*s+0.1)*k for i in range(n)]).reshape(*shape)
def r(t): return [round(float(v),4) for v in t.detach().reshape(-1).tolist()[:6]]
def tryit(name, f):
    try: print(name, f())
    except Exception as e: print(name, 'FAIL', type(e).__name__, str(e)[:150])
img = det((3,1,8,8),1.0,7)
c = nn.Conv2d(1,2,3,padding=1); c.load_state_dict({'weight':det((2,1,3,3),.5,9),'bias':det((2,),.5,10)})
tryit('conv_fwd', lambda: r(c(img)))
tryit('conv_bwd', lambda: (c(img).sum().backward(), r(c.weight.grad))[1])
c.zero_grad() if hasattr(c,'zero_grad') else None
tryit('pool_fwd', lambda: r(F.max_pool2d(c(img),2)))
def pb():
    c2 = nn.Conv2d(1,2,3,padding=1); c2.load_state_dict({'weight':det((2,1,3,3),.5,9),'bias':det((2,),.5,10)})
    F.max_pool2d(torch.relu(c2(img)),2).sum().backward(); return r(c2.weight.grad)
tryit('pool_bwd', pb)
def fl():
    c2 = nn.Conv2d(1,2,3,padding=1); c2.load_state_dict({'weight':det((2,1,3,3),.5,9),'bias':det((2,),.5,10)})
    z = torch.flatten(F.max_pool2d(c2(img),2),1); F.cross_entropy(z[:, :10], torch.tensor([3,1,7])).backward(); return r(c2.weight.grad)
tryit('flatten_ce_bwd', fl)
x = det((1,7,4),1.0,40)
def mha(mask):
    m = nn.MultiheadAttention(4,2,batch_first=True)
    sd = {k: det(tuple(v.shape),.5,41+i) for i,(k,v) in enumerate(sorted(m.state_dict().items()))}
    m.load_state_dict(sd); o,w = m(x,x,x,attn_mask=mask); return r(w[0,3])
tryit('mha_nomask', lambda: mha(None))
tryit('mha_boolmask', lambda: mha(torch.triu(torch.ones(7,7,dtype=torch.bool),diagonal=1)))
tryit('mha_floatmask', lambda: mha(torch.triu(torch.full((7,7),float('-inf')),diagonal=1)))
tryit('masked_fill_softmax', lambda: r(torch.softmax(det((7,7)).masked_fill(torch.triu(torch.ones(7,7,dtype=torch.bool),diagonal=1), float('-inf')),dim=-1)[3]))
tryit('sdpa_causal', lambda: r(F.scaled_dot_product_attention(det((1,7,4)),det((1,7,4),1,2),det((1,7,4),1,3),is_causal=True)))
tryit('matmul_op', lambda: r(det((2,3)) @ det((3,2))))
tryit('t_method', lambda: r(det((2,3)).t()))
tryit('numel', lambda: det((2,3)).numel())
tryit('nn_init', lambda: (lambda l: (nn.init.zeros_(l.weight), r(l.weight))[1])(nn.Linear(2,2)))
tryit('manual_seed', lambda: (torch.manual_seed(0), r(torch.randn(3)), torch.manual_seed(0), r(torch.randn(3)))[1::2])
tryit('one_hot_argmax', lambda: (r(F.one_hot(torch.tensor([0,2]),3)), int(torch.argmax(torch.tensor([1.,5.,2.])))))
tryit('item_inplace_grad', lambda: (lambda w: (w.sum().backward(), r(w.grad))[1])(torch.ones(3, requires_grad=True)))
