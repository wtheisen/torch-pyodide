from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.modules import Module, Parameter
from torch.tensor_factories_ops import tensor_from_data


class RNNBase(Module):
    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False) -> None:
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self._gate_size: int
        self._set_gate_size()

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                gs = self._gate_size
                suffix = "_reverse" if direction == 1 else ""
                k = 1.0 / math.sqrt(hidden_size)  # ai2ai: PyTorch init, U(-1/sqrt(H), 1/sqrt(H))
                w_ih = Parameter(torch.rand((gs, layer_input_size)) * (2 * k) - k)
                w_hh = Parameter(torch.rand((gs, hidden_size)) * (2 * k) - k)
                setattr(self, f"weight_ih_l{layer}{suffix}", w_ih)
                setattr(self, f"weight_hh_l{layer}{suffix}", w_hh)
                if bias:
                    b_ih = Parameter(torch.rand((gs,)) * (2 * k) - k)
                    b_hh = Parameter(torch.rand((gs,)) * (2 * k) - k)
                    setattr(self, f"bias_ih_l{layer}{suffix}", b_ih)
                    setattr(self, f"bias_hh_l{layer}{suffix}", b_hh)

    def _set_gate_size(self) -> None:
        self._gate_size = 4 * self.hidden_size if self.mode in ("LSTM",) else self.hidden_size
        if self.mode == "GRU":
            self._gate_size = 3 * self.hidden_size

    def _param(self, base: str, layer: int, direction: int = 0) -> Tensor:
        suffix = "_reverse" if direction == 1 else ""
        return getattr(self, f"{base}_l{layer}{suffix}")

    def _lstm_step(self, x: Tensor, h: Tensor, c: Tensor,
                   w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> tuple[Tensor, Tensor]:
        gates = x.matmul(w_ih.T) + b_ih + h.matmul(w_hh.T) + b_hh
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def _gru_step(self, x: Tensor, h: Tensor,
                  w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> Tensor:
        gates_x = x.matmul(w_ih.T) + b_ih
        gates_h = h.matmul(w_hh.T) + b_hh
        r_z = gates_x[:, :2 * self.hidden_size] + gates_h[:, :2 * self.hidden_size]
        r = torch.sigmoid(r_z[:, :self.hidden_size])
        z = torch.sigmoid(r_z[:, self.hidden_size:])
        n = torch.tanh(gates_x[:, 2 * self.hidden_size:] + r * gates_h[:, 2 * self.hidden_size:])
        h_new = (1.0 - z) * n + z * h
        return h_new

    def _rnn_relu_step(self, x: Tensor, h: Tensor,
                       w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> Tensor:
        return (x.matmul(w_ih.T) + b_ih + h.matmul(w_hh.T) + b_hh).relu()

    def _rnn_tanh_step(self, x: Tensor, h: Tensor,
                       w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> Tensor:
        return (x.matmul(w_ih.T) + b_ih + h.matmul(w_hh.T) + b_hh).tanh()


    # ai2ai: layer-major, PyTorch-ordered implementation shared by RNN/LSTM/GRU.
    # Output at t is cat(forward_t, backward_t); h_n is ordered
    # [layer0_fwd, layer0_bwd, layer1_fwd, ...] like PyTorch.
    def _run(self, x: Tensor, hx, step, has_c: bool):
        unbatched = len(x.shape) == 2
        if unbatched:
            x = x.unsqueeze(1)
            if hx is not None:
                hx = tuple(h.unsqueeze(1) for h in hx) if has_c else hx.unsqueeze(1)
        elif self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch, _ = x.shape
        n = self.num_layers * self.num_directions
        if hx is None:
            h0 = [torch.zeros((batch, self.hidden_size)) for _ in range(n)]
            c0 = [torch.zeros((batch, self.hidden_size)) for _ in range(n)] if has_c else None
        else:
            hh, cc = hx if has_c else (hx, None)
            h0 = [hh[i] for i in range(n)]
            c0 = [cc[i] for i in range(n)] if has_c else None
        inputs = [x[t] for t in range(seq_len)]
        h_out, c_out = [], []
        for layer in range(self.num_layers):
            dir_outputs = []
            for d in range(self.num_directions):
                idx = layer * self.num_directions + d
                h = h0[idx]
                c = c0[idx] if has_c else None
                w_ih, w_hh = self._param("weight_ih", layer, d), self._param("weight_hh", layer, d)
                if self.bias:
                    b_ih, b_hh = self._param("bias_ih", layer, d), self._param("bias_hh", layer, d)
                else:
                    b_ih = b_hh = torch.zeros((self._gate_size,))
                order = range(seq_len) if d == 0 else range(seq_len - 1, -1, -1)
                outs = [None] * seq_len
                for t in order:
                    if has_c:
                        h, c = step(inputs[t], h, c, w_ih, w_hh, b_ih, b_hh)
                    else:
                        h = step(inputs[t], h, w_ih, w_hh, b_ih, b_hh)
                    outs[t] = h
                dir_outputs.append(outs)
                h_out.append(h)
                if has_c:
                    c_out.append(c)
            if self.num_directions == 1:
                inputs = dir_outputs[0]
            else:
                inputs = [torch.cat([dir_outputs[0][t], dir_outputs[1][t]], dim=-1) for t in range(seq_len)]
            if self.dropout > 0 and self.training and layer < self.num_layers - 1:
                inputs = [torch.nn.functional.dropout(v, self.dropout, True) for v in inputs]
        out = torch.stack(inputs, dim=0)
        h_n = torch.stack(h_out, dim=0)
        c_n = torch.stack(c_out, dim=0) if has_c else None
        if unbatched:
            out = out.squeeze(1)
            h_n = h_n.squeeze(1)
            c_n = c_n.squeeze(1) if has_c else None
        elif self.batch_first:
            out = out.transpose(0, 1)
        return (out, (h_n, c_n)) if has_c else (out, h_n)


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("mode", None)
        super().__init__("LSTM", *args, **kwargs)

    def forward(self, x: Tensor, hx: tuple[Tensor, Tensor] | None = None):
        return self._run(x, hx, self._lstm_step, True)


class GRU(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("mode", None)
        super().__init__("GRU", *args, **kwargs)

    def forward(self, x: Tensor, hx: Tensor | None = None):
        return self._run(x, hx, self._gru_step, False)


class RNN(RNNBase):
    def __init__(self, *args, nonlinearity: str = "tanh", **kwargs) -> None:
        kwargs.pop("mode", None)
        super().__init__("RNN_TANH" if nonlinearity == "tanh" else "RNN_RELU", *args, **kwargs)
        self.nonlinearity = nonlinearity

    def forward(self, x: Tensor, hx: Tensor | None = None):
        step = self._rnn_tanh_step if self.nonlinearity == "tanh" else self._rnn_relu_step
        return self._run(x, hx, step, False)
