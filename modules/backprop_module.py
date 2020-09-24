"""
PyTorch modules that have a custom backward pass. The forward pass is
re-implemented in NumPy, since the modules here do not expose the stored
results of the forward pass.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from scipy.special import expit
from torch import nn


class BackpropModuleMixin(ABC):
    """
    The general interface for modules with custom backward passes. When
    a module is in "attribution mode," the forward and backward
    functions are replaced with the custom functions (re-)implemented in
    NumPy.
    """

    def __init__(self, *args, **kwargs):
        super(BackpropModuleMixin, self).__init__(*args, **kwargs)
        self.attr_mode = False  # Am I in attribution mode?
        self._input = None  # Forward pass input
        self._output = None  # Forward pass output
        self._state = None  # Forward pass stored computations

    def train(self, *args, **kwargs):
        super(BackpropModuleMixin, self).train(*args, **kwargs)
        self.attr_mode = False
        self._input = None
        self._output = None
        self._state = None

    def eval(self):
        super(BackpropModuleMixin, self).eval()
        self.attr_mode = False
        self._input = None
        self._output = None
        self._state = None

    def attr(self):
        """
        Puts the module in attribution mode.
        """
        self.attr_mode = True

    _convert_attr_input_to_numpy = True

    def __call__(self, *args, **kwargs):
        if self.attr_mode:
            if self._convert_attr_input_to_numpy:
                args = list(args)
                for i in range(len(args)):
                    if isinstance(args[i], torch.Tensor):
                        args[i] = args[i].detach().numpy()

                for k in kwargs:
                    if isinstance(kwargs[k], torch.Tensor):
                        kwargs[k] = kwargs[k].detach().numpy()

            return self.attr_forward(*args, **kwargs)
        return super(BackpropModuleMixin, self).__call__(*args, **kwargs)

    def backward(self, *args, **kwargs):
        if self.attr_mode:
            args = list(args)
            for i in range(len(args)):
                if isinstance(args[i], torch.Tensor):
                    args[i] = args[i].detach().numpy()

            for k in kwargs:
                if isinstance(kwargs[k], torch.Tensor):
                    kwargs[k] = kwargs[k].detach().numpy()

            return self.attr_backward(*args, **kwargs)
        self.backward(*args, **kwargs)

    @abstractmethod
    def attr_forward(self, *args, **kwargs):
        """
        The custom forward pass in NumPy.
        """
        raise NotImplementedError("attr_forward not implemented")

    @abstractmethod
    def attr_backward(self, *args, **kwargs):
        """
        The custom backward pass in NumPy.
        """
        raise NotImplementedError("attr_backward not implemented")


class BackpropLinear(BackpropModuleMixin, nn.Linear):
    """
    An interface for nn.Linear.
    """

    def attr_forward(self, x: np.ndarray):
        self._input = [x]
        wx = x @ self.weight.detach().numpy().T
        self._state = dict(wx=wx)
        self._output = wx + self.bias.detach().numpy()
        return self._output


class BackpropRNNMixin(BackpropModuleMixin):
    """
    An interface for PyTorch RNNs in general.
    """

    def __init__(self, *args, **kwargs):
        super(BackpropRNNMixin, self).__init__(*args, **kwargs,
                                               batch_first=True)

    def attr_forward(self, x: np.ndarray):
        """
        Computes the RNN forward pass for all layers and directions. The
        mathematical calculations are defined in the abstract helper
        function _layer_forward.

        :param x: An input to the RNN, of shape (batch_size, seq_len,
            input_size)
        :return: The RNN output, of shape (batch_size, seq_len,
            hidden_size)
        """
        curr_input = x
        self._input = [None] * self.num_layers
        self._state = dict(ltr=[None] * self.num_layers)
        if self.bidirectional:
            self._state["rtl"] = [None] * self.num_layers

        for l in range(self.num_layers):
            self._layer_forward(curr_input, l, 0)
            if self.bidirectional:
                self._layer_forward(np.flip(curr_input, 1), l, 1)
                h_rev = np.flip(self._state["rtl"][l][0], 1)
                curr_input = np.concatenate((self._state["ltr"][l][0], h_rev),
                                            -1)
            else:
                curr_input = self._state["ltr"][l][0]

        self._output = curr_input
        return curr_input

    @abstractmethod
    def _layer_forward(self, x: np.ndarray, layer: int, direction: int):
        """
        This helper function computes the forward pass for a particular
        layer and direction.

        :param x: The input to the layer
        :param layer: The layer number
        :param direction: The direction number (0 for left to right, 1
            for right to left)
        :return: None, but the result should be stored in
            self._state["ltr"][layer] or self._state["rtl"][layer]
        """
        raise NotImplementedError("_layer_forward not implemented")

    num_gates = None

    def _params_numpy(self, prefix: str, layer: int, direction: int) \
            -> Tuple[np.ndarray, ...]:
        """
        Retrieves weight matrices or bias vectors for a particular layer
        and direction.

        :param prefix: "weight_ih" for input weights, "weight_hh" for
            hidden state weights, "bias_ih" for input biases, or
            "bias_hh" for hidden state biases
        :param layer: The layer to retrieve weights for
        :param direction: The direction to retrieve weights for
        :return: The weight/bias matrices
        """
        p = prefix + "_l" + str(layer) + ("_reverse" if direction == 1 else "")
        return np.split(getattr(self, p).detach().numpy(), self.num_gates)


class BackpropLSTM(BackpropRNNMixin, nn.LSTM):
    """
    An interface for nn.LSTM.
    """

    num_gates = 4

    def _layer_forward(self, x: np.ndarray, layer: int, direction: int):
        if direction == 0:
            self._input[layer] = x

        batch_size, seq_len, _ = x.shape
        x = x[:, :, :, np.newaxis]

        # Get parameters
        kwargs = {"layer": layer, "direction": direction}
        w_ii, w_if, w_ig, w_io = self._params_numpy("weight_ih", **kwargs)
        w_hi, w_hf, w_hg, w_ho = self._params_numpy("weight_hh", **kwargs)
        biases_i = self._params_numpy("bias_ih", **kwargs)
        biases_h = self._params_numpy("bias_hh", **kwargs)
        b_ii, b_if, b_ig, b_io = [b[:, np.newaxis] for b in biases_i]
        b_hi, b_hf, b_hg, b_ho = [b[:, np.newaxis] for b in biases_h]

        # Initialize
        h = np.zeros((batch_size, seq_len, self.hidden_size))
        i = np.zeros((batch_size, seq_len, self.hidden_size))
        f = np.zeros((batch_size, seq_len, self.hidden_size))
        g_pre = np.zeros((batch_size, seq_len, self.hidden_size))
        g = np.zeros((batch_size, seq_len, self.hidden_size))
        o = np.zeros((batch_size, seq_len, self.hidden_size))
        c = np.zeros((batch_size, seq_len, self.hidden_size))

        # Forward pass
        h_prev = np.zeros((batch_size, self.hidden_size, 1))
        c_prev = np.zeros((batch_size, self.hidden_size))
        for t in range(seq_len):
            i_temp = (w_ii @ x[:, t] + b_ii + w_hi @ h_prev + b_hi).squeeze(-1)
            f_temp = (w_if @ x[:, t] + b_if + w_hf @ h_prev + b_hf).squeeze(-1)
            g_temp = (w_ig @ x[:, t] + b_ig + w_hg @ h_prev + b_hg).squeeze(-1)
            o_temp = (w_io @ x[:, t] + b_io + w_ho @ h_prev + b_ho).squeeze(-1)

            i[:, t] = expit(i_temp)
            f[:, t] = expit(f_temp)
            g_pre[:, t] = g_temp
            g[:, t] = np.tanh(g_temp)
            o[:, t] = expit(o_temp)

            c[:, t] = f[:, t] * c_prev + i[:, t] * g[:, t]
            h[:, t] = o[:, t] * np.tanh(c[:, t])

            h_prev = h[:, t, :, np.newaxis]
            c_prev = c[:, t]

        # Save trace to state
        if direction == 0:
            self._state["ltr"][layer] = h, c, i, f, g, g_pre, w_ig.T, w_hg.T
        else:
            self._state["rtl"][layer] = h, c, i, f, g, g_pre, w_ig.T, w_hg.T


class BackpropGRU(BackpropRNNMixin, nn.GRU):
    """
    An interface for nn.GRU.
    """

    num_gates = 3

    def _layer_forward(self, x: np.ndarray, layer: int, direction: int):
        if direction == 0:
            self._input[layer] = x

        batch_size, seq_len, _ = x.shape
        x = x[:, :, :, np.newaxis]

        # Get parameters
        kwargs = {"layer": layer, "direction": direction}
        w_ir, w_iz, w_in = self._params_numpy("weight_ih", **kwargs)
        w_hr, w_hz, w_hn = self._params_numpy("weight_hh", **kwargs)
        biases_i = self._params_numpy("bias_ih", **kwargs)
        biases_h = self._params_numpy("bias_hh", **kwargs)
        b_ir, b_iz, b_in = [b[:, np.newaxis] for b in biases_i]
        b_hr, b_hz, b_hn = [b[:, np.newaxis] for b in biases_h]

        # Initialize
        h = np.zeros((batch_size, seq_len, self.hidden_size))
        r = np.zeros((batch_size, seq_len, self.hidden_size))
        z = np.zeros((batch_size, seq_len, self.hidden_size))
        n_pre = np.zeros((batch_size, seq_len, self.hidden_size))
        n = np.zeros((batch_size, seq_len, self.hidden_size))

        # Forward pass
        h_prev = np.zeros((batch_size, self.hidden_size, 1))
        for t in range(seq_len):
            r_temp = (w_ir @ x[:, t] + b_ir + w_hr @ h_prev + b_hr).squeeze(-1)
            z_temp = (w_iz @ x[:, t] + b_iz + w_hz @ h_prev + b_hz).squeeze(-1)
            r[:, t] = expit(r_temp)
            z[:, t] = expit(z_temp)

            n_pre_i = (w_in @ x[:, t] + b_in).squeeze(-1)
            n_pre_h = (w_hn @ h_prev + b_hn).squeeze(-1)
            n_pre[:, t] = n_pre_i + r[:, t] * n_pre_h
            n[:, t] = np.tanh(n_pre[:, t])

            h[:, t] = (1 - z[:, t]) * n[:, t] + z[:, t] * h_prev.squeeze(-1)
            h_prev = h[:, t, :, np.newaxis]

            # Save trace to state
            if direction == 0:
                self._state["ltr"][layer] = h, r, z, n, n_pre, w_in.T, w_hn.T
            else:
                self._state["rtl"][layer] = h, r, z, n, n_pre, w_in.T, w_hn.T


class BackpropLayerNorm(BackpropModuleMixin, nn.LayerNorm):
    """
    Layer normalization for the Transformer.
    """

    def attr_forward(self, x):
        axes = tuple(range(-1, -len(self.normalized_shape) - 1, -1))
        num = x - x.mean(axis=axes, keepdims=True)
        den = np.sqrt(x.var(axis=axes, keepdims=True) + self.eps)
        if not self.elementwise_affine:
            return num / den

        gamma = self.weight.detach().numpy()
        beta = self.bias.detach().numpy()
        return (num / den) * gamma + beta
