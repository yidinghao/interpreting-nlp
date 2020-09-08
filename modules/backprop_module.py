"""
PyTorch modules have a custom backward pass.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from scipy.special import expit
from torch import nn


class BackpropModuleMixin(ABC):
    """
    An interface for implementing custom backward passes.
    """

    def __init__(self, *args, **kwargs):
        super(BackpropModuleMixin, self).__init__(*args, **kwargs)
        self._attr = False  # Attribution mode or not
        self._input = None
        self._output = None
        self._state = None  # Forward pass stored computations

    def train(self, *args, **kwargs):
        """
        Puts the module in training mode, and takes it out of
        attribution mode
        """
        super(BackpropModuleMixin, self).train(*args, **kwargs)
        self._attr = False
        self._state = None

    def eval(self, attr: bool = False):
        """
        Puts the module in evaluation mode, and optionally attribution
        mode

        :param attr: Whether or not to put the module in attribution
            mode
        """
        super(BackpropModuleMixin, self).eval()
        self._attr = attr
        if not attr:
            self._state = None

    def attr(self):
        self.eval(attr=True)

    def __call__(self, *args, **kwargs):
        if self._attr:
            args = list(args)
            for i in range(len(args)):
                if isinstance(args[i], torch.Tensor):
                    args[i] = args[i].detach().numpy()

            self._input = args
            self._output = self.attr_forward(*args, **kwargs)
            return self._output
        return super(BackpropModuleMixin, self).__call__(*args, **kwargs)

    @abstractmethod
    def attr_forward(self, *args, **kwargs):
        raise NotImplementedError("Please implement the custom forward pass")

    @abstractmethod
    def attr_backward(self, *args, **kwargs):
        raise NotImplementedError("Please implement the custom backward pass")


class BackpropLinear(BackpropModuleMixin, nn.Linear):
    """
    Linear module
    """

    def attr_forward(self, x: np.ndarray):
        """
        Forward pass.

        :param x:
        :return:
        """
        wx = x @ self.weight.detach().numpy().T
        self._state = dict(wx=wx)
        return wx + self.bias.detach().numpy()


class BackpropLSTM(BackpropModuleMixin, nn.LSTM):
    """
    LSTM module
    """

    def __init__(self, *args, **kwargs):
        super(BackpropLSTM, self).__init__(*args, **kwargs, batch_first=True)

    def attr_forward(self, x: np.ndarray):
        """
        Computes the LSTM forward pass using NumPy operations and saves
        a trace of the computation to self.traces and self.traces_rev.

        :param x: An input to the LSTM (batch_size, seq_len, input_size)
        :return: The LSTM output
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

    def _layer_forward(self, x: np.ndarray, layer: int, direction: int):
        """
        Performs a forward pass using numpy operations for one layer.

        :param x: An input of shape (batch_size, seq_len, input_size)
        :param layer: The layer to perform the forward pass for
        :param direction: The direction to perform the forward pass for

        :return: None, but saves a trace of the forward pass, consisting
            of the weights used and the values of the gates, to
            self._traces or self._traces_rev
        """
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
            i_temp = w_ii @ x[:, t] + b_ii + w_hi @ h_prev + b_hi
            f_temp = w_if @ x[:, t] + b_if + w_hf @ h_prev + b_hf
            g_temp = w_ig @ x[:, t] + b_ig + w_hg @ h_prev + b_hg
            o_temp = w_io @ x[:, t] + b_io + w_ho @ h_prev + b_ho

            i[:, t] = expit(i_temp[:, :, 0])
            f[:, t] = expit(f_temp[:, :, 0])
            g_pre[:, t] = g_temp[:, :, 0]
            g[:, t] = np.tanh(g_pre[:, t])
            o[:, t] = expit(o_temp[:, :, 0])

            c[:, t] = f[:, t] * c_prev + i[:, t] * g[:, t]
            h[:, t] = o[:, t] * np.tanh(c[:, t])

            h_prev = h[:, t, :, np.newaxis]
            c_prev = c[:, t]

        # Save trace to state
        if direction == 1:
            self._state["rtl"][layer] = h, c, i, f, g, g_pre, w_ig, w_hg
        else:
            self._input[layer] = x
            self._state["ltr"][layer] = h, c, i, f, g, g_pre, w_ig, w_hg

    def _params_numpy(self, prefix: str, layer: int, direction: int = 0) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves weight matrices or bias vectors for a particular layer
        and direction.

        :param prefix: "weight_ih" for input weights, "weight_hh" for
            hidden state weights, "bias_ih" for input biases, or
            "bias_hh" for hidden state biases

        :param layer: The layer to retrieve weights for

        :param direction: The direction to retrieve weights for
        :return:
        """
        n = self.hidden_size
        p = prefix + "_l" + str(layer)
        if direction == 1:
            p += "_reverse"

        w = getattr(self, p).detach().numpy()
        return tuple(w[n * i:n * (i + 1)] for i in range(4))
