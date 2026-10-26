"""A small torchvision for the browser build: just the transforms the AI2AI book's code uses
(Compose, ToTensor, Normalize, Lambda). No datasets or models (they need downloads and C code)."""
from . import transforms

__version__ = "0.0.1+ai2ai"
