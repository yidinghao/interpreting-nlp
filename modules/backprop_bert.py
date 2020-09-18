from typing import Tuple, Union

import numpy as np
import scipy.special as sp
from torch import nn
from transformers import modeling_bert as bert

from modules import backprop_module as bp

Arrays = Union[np.ndarray, Tuple[np.ndarray, ...]]
NormalLayer = Union[nn.Linear, nn.LayerNorm]
BackpropLayer = Union[bp.BackpropLinear, bp.BackpropLayerNorm]

_erf_approx = lambda x: np.tanh(np.sqrt(2. / np.pi) * (x + 0.044715 * x ** 3))
activations = dict(relu=lambda x: np.maximum(x, 0.),
                   gelu=lambda x: x * .5 * (1. + sp.erf(x / np.sqrt(2.))),
                   swish=lambda x: x * sp.expit(x),
                   gelu_new=lambda x: x * .5 * _erf_approx(x),
                   mish=None)


class BackpropBertMixin(bp.BackpropModuleMixin):
    """
    Interface for BERT modules with custom backprop.
    """

    _layer_types = dict(linear=bp.BackpropLinear,
                        layernorm=bp.BackpropLayerNorm)

    @classmethod
    def convert_to_attr(cls, layer: NormalLayer) -> BackpropLayer:
        if isinstance(layer, nn.Linear):
            linear_class = cls._layer_types["linear"]
            new_layer = linear_class(layer.in_features, layer.out_features)
        elif isinstance(layer, nn.LayerNorm):
            ln_class = cls._layer_types["layernorm"]
            new_layer = ln_class(layer.normalized_shape, eps=layer.eps,
                                 elementwise_affine=layer.elementwise_affine)
        else:
            raise TypeError("Only linear and layernorm can be converted")

        new_layer.load_state_dict(layer.state_dict())
        return new_layer


class BackpropBertSelfAttention(BackpropBertMixin, bert.BertSelfAttention):
    """
    A BERT self-attention module.
    """

    def __init__(self, *args, **kwargs):
        super(BackpropBertSelfAttention, self).__init__(*args, **kwargs)
        self.query = BackpropBertSelfAttention.convert_to_attr(self.query)
        self.key = BackpropBertSelfAttention.convert_to_attr(self.key)
        self.value = BackpropBertSelfAttention.convert_to_attr(self.value)

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
        attention_scores /= np.sqrt(self.attention_head_size)
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


class BackpropBertSelfOutput(BackpropBertMixin, bert.BertSelfOutput):
    """
    BERT self-output module
    """

    def __init__(self, config):
        super(BackpropBertSelfOutput, self).__init__(config)
        self.dense = BackpropBertSelfOutput.convert_to_attr(self.dense)
        self.LayerNorm = BackpropBertSelfOutput.convert_to_attr(self.LayerNorm)

    def attr(self):
        super(BackpropBertSelfOutput, self).attr()
        self.dense.attr()
        self.LayerNorm.attr()

    def attr_forward(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class BackpropBertIntermediate(BackpropBertMixin, bert.BertIntermediate):
    """
    Bert Intermediate
    """

    def __init__(self, config):
        super(BackpropBertIntermediate, self).__init__(config)
        self.dense = BackpropBertMixin.convert_to_attr(self.dense)
        self.intermediate_act_fn_numpy = activations[config.hidden_act]

    def attr(self):
        super(BackpropBertIntermediate, self).attr()
        self.dense.attr()

    def attr_forward(self, hidden_states):
        return self.intermediate_act_fn_numpy(self.dense(hidden_states))


class BackpropBertOutput(BackpropBertMixin, bert.BertOutput):
    """
    Bert Output
    """

    def __init__(self, config):
        super(BackpropBertOutput, self).__init__(config)
        self.dense = BackpropBertOutput.convert_to_attr(self.dense)
        self.LayerNorm = BackpropBertOutput.convert_to_attr(self.LayerNorm)

    def attr(self):
        super(BackpropBertOutput, self).attr()
        self.dense.attr()
        self.LayerNorm.attr()

    def attr_forward(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class BackpropBertAttention(bp.BackpropModuleMixin, bert.BertAttention):
    def attr_forward(self, hidden_states, **kwargs):
        self_outputs = self.self(hidden_states, **kwargs)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]
