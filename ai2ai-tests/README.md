# ai2ai comparison tests

Each script runs unchanged under real PyTorch and under this fork in Pyodide, and the outputs must match.

- `python3 suite.py > real.txt` (real PyTorch)
- Serve a folder with Pyodide, `runtime.mjs` and a tarball of `python/torch`, using `page.html` as the page, then `python3 run.py suite.py page.html > fork.txt` (headless Chromium with SwiftShader WebGPU).
- `diff real.txt fork.txt`: only timings, `manual_seed` streams and `torch.Size` printing may differ.
