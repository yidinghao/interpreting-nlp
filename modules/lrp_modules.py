import numpy as np

import modules.backprop_module as backprop
from attribution.lrp import lrp_linear


class LRPLinear(backprop.BackpropLinear):
    """
    A Linear module with LRP.
    """

    def attr_backward(self, rel_y: np.ndarray,
                      eps: float = 0.001) -> np.ndarray:
        return lrp_linear(self._input[0], self._state["wx"], rel_y,
                          self.weight.detach().numpy().T, eps=eps)


class LRPLSTM(backprop.BackpropLSTM):
    """
    An LSTM module with LRP.
    """

    def attr_backward(self, rel_y: np.ndarray,
                      eps: float = 0.001) -> np.ndarray:
        """
        Computes the LRP backward pass using NumPy operations based on
        the most recent call of self.lrp_forward.

        :param rel_y: The relevance of the LSTM output (seq_len,
            hidden_size)

        :param eps: The LRP stabilizer

        :return: The relevance of the input from the most recent
        call of self.lrp_forward
        """
        if self.bidirectional:
            curr_rel, curr_rel_rev = np.split(rel_y, 2, axis=-1)
            curr_rel_rev = np.flip(curr_rel_rev, 1)
        else:
            curr_rel = rel_y
            curr_rel_rev = None

        for l in reversed(range(self.num_layers)):
            rel_x = self._layer_backward(curr_rel, l, 0, eps=eps)
            if self.bidirectional:
                rel_x_rev = self._layer_backward(curr_rel_rev, l, 1, eps=eps)
                rel_x += np.flip(rel_x_rev, 1)

            if self.bidirectional and l > 0:
                curr_rel = rel_x[:, :, :self.hidden_size]
                curr_rel_rev = np.flip(rel_x[:, :, self.hidden_size:], 1)
            else:
                curr_rel = rel_x

        return curr_rel

    def _layer_backward(self, rel_y: np.ndarray, layer: int, direction: int,
                        eps: float = 0.001) -> np.ndarray:
        """
        Performs a backward pass using numpy operations for one layer.

        :param rel_y: The relevance flowing to this layer

        :param layer: The layer to perform the backward pass for

        :param direction: The direction to perform the backward pass for

        :return: The relevance of the layer inputs
        """
        if direction == 0:
            x = self._input[layer]
            h, c, i, f, g, g_pre, w_ig, w_hg = self._state["ltr"][layer]
        else:
            x = np.flip(self._input[layer], 1)
            h, c, i, f, g, g_pre, w_ig, w_hg = self._state["rtl"][layer]

        batch_size, seq_len, _ = x.shape

        # Initialize
        rel_h = np.zeros((batch_size, seq_len + 1, self.hidden_size))
        rel_c = np.zeros((batch_size, seq_len + 1, self.hidden_size))
        rel_g = np.zeros(g.shape)
        rel_x = np.zeros(x.shape)

        # Backward pass
        rel_h[:, 1:] = rel_y
        for t in reversed(range(seq_len)):
            rel_c[:, t + 1] += rel_h[:, t + 1]
            rel_c[:, t] = lrp_linear(f[:, t] * c[:, t - 1], c[:, t],
                                     rel_c[:, t + 1], eps=eps)
            rel_g[:, t] = lrp_linear(i[:, t] * g[:, t], c[:, t],
                                     rel_c[:, t + 1], eps=eps)
            rel_x[:, t] = lrp_linear(x[:, t], g_pre[:, t], rel_g[:, t],
                                     w=w_ig, eps=eps)

            h_prev = np.zeros((batch_size, self.hidden_size)) if t == 0 \
                else h[:, t - 1]
            rel_h[:, t] += lrp_linear(h_prev, g_pre[:, t], rel_g[:, t], w=w_hg,
                                      eps=eps)

        self._rel_h = rel_h
        self._rel_c = rel_c
        return rel_x