from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._tensor import Tensor


def getitem_from_tensor(tensor: "Tensor", key: object) -> object:
    """ai2ai: PyTorch-style basic indexing (int, slice, ..., None) plus one integer
    tensor/list index per dim (index_select) and full-shape bool masks."""
    T = tensor.__class__
    if isinstance(key, T) and key._dtype == "bool":
        from .tensor_ops import masked_select_from_tensor
        return masked_select_from_tensor(tensor, key)
    if not isinstance(key, tuple):
        key = (key,)
    n_real = sum(1 for k in key if k is not None and k is not Ellipsis)
    if any(k is Ellipsis for k in key):
        i = next(j for j, k in enumerate(key) if k is Ellipsis)
        key = key[:i] + (slice(None),) * max(0, len(tensor._shape) - n_real) + key[i + 1:]
    result = tensor
    dim = 0
    for k in key:
        if k is None:
            result = result.unsqueeze(dim)
            dim += 1
        elif isinstance(k, bool):
            raise TypeError("bool scalar index not supported")
        elif isinstance(k, int):
            size = result._shape[dim]
            result = result.select(dim=dim, index=k + size if k < 0 else k)
        elif isinstance(k, slice):
            if not (k.start is None and k.stop is None and k.step in (None, 1)):
                result = result.slice(dim=dim, start=k.start, end=k.stop, step=k.step or 1)
            dim += 1
        elif isinstance(k, (list, T)):
            import torch
            idx = k if isinstance(k, T) else torch.tensor(k)
            if idx._dtype == "bool":
                raise TypeError("bool tensor index must index the whole tensor")
            ishape = list(idx._shape)
            size = result._shape[dim]
            flat = idx.reshape(-1)
            flat = flat.add(flat.lt(0).to(dtype=flat._dtype).mul(size)) if size else flat
            picked = result.index_select(dim=dim, index=flat)
            if len(ishape) != 1:
                shp = list(picked._shape)
                picked = picked.reshape(shp[:dim] + ishape + shp[dim + 1:])
            result = picked
            dim += len(ishape)
        else:
            raise TypeError(f"Unsupported index type: {type(k)}")
    return result


def _np_key(T, key):
    import numpy as np

    def conv(k):
        if isinstance(k, T):
            return np.array(k.tolist(), dtype=bool if k._dtype == "bool" else np.int64)
        if isinstance(k, list):
            return np.array(k)
        return k
    return tuple(conv(k) for k in key) if isinstance(key, tuple) else conv(key)


def setitem_from_tensor(tensor: "Tensor", key: object, value: object) -> None:
    """ai2ai: t[key] = value, in place, with numpy (= PyTorch) indexing rules.
    With autograd on, the write is recorded: the result is where(written, value, old)."""
    import copy
    import numpy as np
    import torch
    from .autograd import is_grad_enabled

    T = tensor.__class__
    k = _np_key(T, key)
    dt = tensor._dtype
    vt = value if isinstance(value, T) else None
    grad = is_grad_enabled() and (tensor._requires_grad or (vt is not None and vt._requires_grad))
    if grad and tensor._requires_grad and tensor._node is None:
        raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
    if not grad:
        a = np.array(tensor.detach().tolist(), dtype=np.asarray(tensor.detach().numpy()).dtype).reshape(tensor._shape)
        a[k] = np.asarray(vt.detach().numpy()) if vt is not None else value
        tensor._set(torch.tensor(a.tolist() if a.ndim else a.item(), dtype=dt).reshape(list(tensor._shape)))
        return
    shape = list(tensor._shape)
    n = int(np.prod(shape)) if shape else 1
    pos = np.arange(n).reshape(shape)[k]
    pos = np.asarray(pos).reshape(-1)
    m = pos.size
    if m == 0:
        return
    sel_shape = list(np.arange(n).reshape(shape)[k].shape)
    v = vt if vt is not None else torch.tensor(value, dtype=dt)
    if v._dtype != dt:
        v = v.to(dt)
    v = v + torch.zeros(sel_shape, dtype=dt)           # broadcast, grads sum back
    P = np.zeros((n, m))
    P[pos, np.arange(m)] = 1
    placed = (torch.tensor(P.tolist(), dtype=dt) @ v.reshape([m, 1])).reshape(shape)
    mask = np.zeros(n, dtype=bool)
    mask[pos] = True
    old = copy.copy(tensor)
    if old._node is not None:   # the earlier graph now ends at the copy
        old._node.tensor = old
    new = torch.where(torch.tensor(mask.reshape(shape).tolist()), placed, old)
    tensor._set(new)
    tensor._node = new._node
    tensor._requires_grad = new._requires_grad
    if new._node is not None:
        new._node.tensor = tensor
