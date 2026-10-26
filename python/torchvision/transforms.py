"""torchvision.transforms, the pieces that work on arrays and tensors: Compose, ToTensor, Normalize, Lambda."""
import numpy as np
import torch

__all__ = ["Compose", "ToTensor", "Normalize", "Lambda"]


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        inner = "".join("\n    %r" % t for t in self.transforms)
        return "Compose(%s\n)" % inner


class ToTensor:
    """An H x W (x C) image (array or PIL image) to a float C x H x W tensor; 8-bit values are scaled to [0, 1]."""

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic.float()
        a = np.asarray(pic)
        scale = a.dtype == np.uint8
        a = a.astype(np.float32)
        if scale:
            a = a / 255.0
        if a.ndim == 2:
            a = a[None, :, :]
        elif a.ndim == 3:
            a = a.transpose(2, 0, 1)
        return torch.tensor(np.ascontiguousarray(a))

    def __repr__(self):
        return "ToTensor()"


class Normalize:
    """(x - mean) / std, channel by channel, on a C x H x W (or N x C x H x W) tensor."""

    def __init__(self, mean, std, inplace=False):
        self.mean, self.std = list(mean), list(std)

    def __call__(self, t):
        shape = [1] * len(t.shape)
        shape[-3] = -1
        m = torch.tensor(self.mean, dtype=t.dtype).reshape(shape)
        s = torch.tensor(self.std, dtype=t.dtype).reshape(shape)
        return (t - m) / s

    def __repr__(self):
        return "Normalize(mean=%s, std=%s)" % (tuple(self.mean), tuple(self.std))


class Lambda:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)

    def __repr__(self):
        return "Lambda()"
