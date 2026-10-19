import torch, math
def det(shape, k=1.0, s=0):
    n=1
    for d in shape: n*=d
    return torch.tensor([math.sin(1.3*i+0.7*s+0.1)*k for i in range(n)]).reshape(*shape)
def r(t): v=t.detach().reshape(-1).tolist(); return [round(float(x),4) for x in v[-4:]]+[round(sum(abs(float(x)) for x in v),3)]
cases={'5x1x4@4x12':((5,1,4),(4,12)),'2x5x4@4x12':((2,5,4),(4,12)),'3x2x5x4@4x3':((3,2,5,4),(4,3)),'2x5x4@2x4x3':((2,5,4),(2,4,3)),'4x2x1x3@4x2x3x5':((4,2,1,3),(4,2,3,5)),'2x5x4@1x4x3':((2,5,4),(1,4,3)),'5x4@2x4x3':((5,4),(2,4,3)),'vec':((4,),(4,3))}
for k,(a,b) in cases.items():
    A=det(a,1,1).requires_grad_() if hasattr(torch.Tensor,'requires_grad_') else det(a,1,1); B=det(b,1,2)
    try:
        o=A@B; print(k, list(o.shape), r(o))
    except Exception as e: print(k,'FAIL',str(e)[:80])
A=det((2,1,3,4),1,5).requires_grad_(); B=det((3,4,2),1,6).requires_grad_()
(A@B).pow(2).sum().backward(); print('grad', r(A.grad), r(B.grad))
