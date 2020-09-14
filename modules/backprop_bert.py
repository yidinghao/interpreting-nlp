import math
from typing import Tuple, Union

import numpy as np
import scipy.special as sp
from torch import nn
from transformers import modeling_bert as bert

from modules.backprop_module import BackpropModuleMixin, BackpropLinear
from modules.lrp_modules import LRPLinear

Arrays = Union[np.ndarray, Tuple[np.ndarray, ...]]


def convert_linear_to_attr(linear: nn.Linear) -> BackpropLinear:
    new_layer = LRPLinear(linear.in_features, linear.out_features)
    new_layer.load_state_dict(linear.state_dict())
    return new_layer


class BackpropBertSelfAttention(BackpropModuleMixin, bert.BertSelfAttention):
    """
    A BERT self-attention module.
    """

    def attr(self):
        super(BackpropBertSelfAttention, self).attr()
        self.query.attr()
        self.key.attr()
        self.value.attr()

    def attr_forward(self, hidden_states: np.ndarray,
                     attention_mask: np.ndarray = None,
                     head_mask: np.ndarray = None,
                     encoder_hidden_states: np.ndarray = None,
                     encoder_attention_mask: np.ndarray = None,
                     output_attentions: bool = False) -> Arrays:
        """
        The NumPy forward pass.

        :param hidden_states: The input to the attention layer
        :param attention_mask: The attention mask
        :param head_mask: An optional mask for heads
        :param encoder_hidden_states: The encoder hidden states, passed
            to this layer when used in a decoder
        :param encoder_attention_mask: None
        :param output_attentions: False
        :return:
        """
        mixed_query_layer = self.query(hidden_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self._attr_transpose_for_scores(mixed_query_layer)
        key_layer = self._attr_transpose_for_scores(mixed_key_layer)
        value_layer = self._attr_transpose_for_scores(mixed_value_layer)

        attention_scores = query_layer @ key_layer.transpose(0, 1, 3, 2)
        attention_scores /= math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = sp.softmax(attention_scores, axis=-1)
        if head_mask is not None:
            attention_probs *= head_mask

        context_layer = (attention_probs @ value_layer).transpose(0, 2, 1, 3)
        context_layer = np.reshape(context_layer, context_layer.shape[:-2] + \
                                   (self.all_head_size,))

        if output_attentions:
            return context_layer, attention_probs
        else:
            return context_layer

    def _attr_transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads,
                                      self.attention_head_size)
        x = np.reshape(x, new_x_shape)
        return x.transpose(0, 2, 1, 3)


class BackpropBertSelfOutput(BackpropModuleMixin, bert.BertSelfOutput):
    """
    BERT self-output module
    """

    def attr_forward(self, *args, **kwargs):
        pass
