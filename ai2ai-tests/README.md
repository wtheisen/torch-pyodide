# ai2ai comparison tests

Each script runs unchanged under real PyTorch and under this fork, and the outputs must match.

| script | what |
|---|---|
| `suite.py` | training loops (SGD, Adam), CNN, RNN/LSTM/GRU, MultiheadAttention, encoder layer, DataLoader, nn.Module |
| `api.py` | ~100 API checks: broadcasting, reductions, indexing, views, losses, activations, layers, autograd, optimizers |
| `micro3.py`, `m13.py` | conv/pool backward, masks, matmul broadcasting |
| `perf.py` | timing of a small CNN and a transformer block |
| `opdiff.py` | per-op comparison of the numpy runtime against the WebGPU runtime (runs in the browser) |

Run against real PyTorch: `python3 api.py > real.txt`.

Run the fork on the CPU (numpy) backend, no browser needed:
`TORCH_PYODIDE_BACKEND=cpu PYTHONPATH=../python python3 api.py > fork.txt`, then `diff real.txt fork.txt`.

Run in Pyodide: serve a folder with Pyodide (plus its numpy wheel), a tarball of `python/torch` and `cpu.html`,
then `python3 run.py api.py cpu.html` (CPU) or `run.py api.py "cpu.html?gpu=1&backend=webgpu"` (WebGPU, needs runtime.mjs;
headless Chromium uses SwiftShader).

Known, accepted differences: `torch.manual_seed` streams, dtype/shape printing (`'int64'` vs `torch.int64`,
`(2, 2)` vs `torch.Size`), `-0.0` vs `0.0`. The WebGPU backend still fails a few api.py checks
(its kernels: lerp/addcmul/addcdiv, any/all, argmax without dim, log_softmax over dim 0, nearest upsampling
); the CPU backend is the default.
