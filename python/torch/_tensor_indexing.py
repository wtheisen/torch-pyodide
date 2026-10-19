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
