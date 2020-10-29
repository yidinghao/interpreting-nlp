from typing import List, Tuple, Union

import numpy as np
import scipy.special as sp
import torch
from torch import nn
from transformers import modeling_bert as bert
from transformers.configuration_bert import BertConfig

from modules import backprop_module as bp

# Shorthands for different array sizes
HiddenArray = np.ndarray  # (batch_size, seq_len, hidden_size)
AttentionArray = np.ndarray  # (batch_size, num_heads, seq_len, _)
IndexTensor = torch.LongTensor  # (batch_size, seq_len)
EmbeddingTensor = torch.FloatTensor  # (batch_size, seq_len, hidden_size)

NormalLayer = Union[nn.Linear, nn.LayerNorm]
AttnSubLayer = Union[bert.BertSelfAttention, bert.BertSelfOutput]

_erf_approx = lambda x: np.tanh(np.sqrt(2. / np.pi) * (x + 0.044715 * x ** 3))
activations = dict(relu=lambda x: np.maximum(x, 0.),
                   gelu=lambda x: x * .5 * (1. + sp.erf(x / np.sqrt(2.))),
                   swish=lambda x: x * sp.expit(x),
                   gelu_new=lambda x: x * .5 * _erf_approx(x),
                   mish=None)


def hidden_to_attention(h: HiddenArray, num_heads: int) -> AttentionArray:
    return h.reshape(h.shape[:-1] + (num_heads, -1)).transpose(0, 2, 1, 3)


def attention_to_hidden(a: AttentionArray) -> HiddenArray:
    a = a.transpose(0, 2, 1, 3)
    return a.reshape(a.shape[:-2] + (-1,))


class BackpropBertMixin(bp.BackpropModuleMixin):
    """
    Interface for BERT modules with custom backprop. This mixin
    introduces a function that converts a normal PyTorch module to a
    custom backprop module.
    """

    _layer_types = {nn.Linear: bp.BackpropLinear,
                    nn.LayerNorm: bp.BackpropLayerNorm}
    _bert_layer_types = {bert.BertSelfAttention: None,
                         bert.BertSelfOutput: None,
                         bert.BertAttention: None,
                         bert.BertIntermediate: None,
                         bert.BertOutput: None}

    def convert_to_attr(self, layer: NormalLayer) -> bp.BackpropModuleMixin:
        """
        Converts nn.Linear or nn.LayerNorm to custom backprop layers. In
        order to use this, child classes must override the _layer_types
        dict defined above.

        :param layer: An nn.Linear or nn.LayerNorm module
        :return: The corresponding module with custom backprop
        """
        if isinstance(layer, nn.Linear):
            linear_class = self._layer_types[nn.Linear]
            new_layer = linear_class(layer.in_features, layer.out_features)
        elif isinstance(layer, nn.LayerNorm):
            ln_class = self._layer_types[nn.LayerNorm]
            new_layer = ln_class(layer.normalized_shape, eps=layer.eps,
                                 elementwise_affine=layer.elementwise_affine)
        else:
            raise TypeError("Cannot convert layer of type " + str(type(layer)))

        new_layer.load_state_dict(layer.state_dict())
        return new_layer

    def convert_bert_to_attr(self, layer: AttnSubLayer, config: BertConfig):
        new_layer = self._bert_layer_types[type(layer)](config)
        new_layer.load_state_dict(layer.state_dict())
        return new_layer

    def hidden_to_attention(self, h: HiddenArray) -> AttentionArray:
        return hidden_to_attention(h, self.num_attention_heads)

    @staticmethod
    def attention_to_hidden(a: AttentionArray) -> HiddenArray:
        return attention_to_hidden(a)


class BackpropBertEmbeddings(BackpropBertMixin, bert.BertEmbeddings):
    """
    Combines word embeddings with positional embeddings and token type
    embeddings.
    """

    _convert_attr_input_to_numpy = False

    def __init__(self, config: BertConfig):
        super(BackpropBertEmbeddings, self).__init__(config)
        self.LayerNorm = self.convert_to_attr(self.LayerNorm)

    def attr(self):
        super(BackpropBertEmbeddings, self).attr()
        self.LayerNorm.attr()

    def attr_forward(self, input_ids: IndexTensor = None,
                     inputs_embeds: EmbeddingTensor = None,
                     token_type_ids: IndexTensor = None,
                     position_ids: IndexTensor = None) -> HiddenArray:
        """
        Adds the word embeddings to positional and token type
        embeddings.

        :param input_ids: Indices for an input sequence
        :param inputs_embeds: Embeddings for an input sequence. Either
            this or input_ids must be none
        :param token_type_ids: idk what this is
        :param position_ids: Positions
        :return: The input to the first BERT layer
        """
        assert (input_ids is None) != (inputs_embeds is None)

        if input_ids is not None:
            input_shape = input_ids.shape
            inputs_embeds = self.word_embeddings(input_ids).detach().numpy()
        else:
            input_shape = inputs_embeds.shape[:-1]
            inputs_embeds = inputs_embeds.detach().numpy()

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        position_embeds = self.position_embeddings(position_ids)
        position_embeds = position_embeds.detach().numpy()

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long,
                                         device=self.position_ids.device)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        token_type_embeds = token_type_embeds.detach().numpy()

        self._state = inputs_embeds, position_embeds, token_type_embeds
        return self.LayerNorm(inputs_embeds + position_embeds +
                              token_type_embeds)


class BackpropBertSelfAttention(BackpropBertMixin, bert.BertSelfAttention):
    """
    A BERT self-attention module. This module is responsible for
    implementing the scaled dot-product attention equation. This module
    is combined with BackpropBertSelfOutput to form an attention layer.
    """

    def __init__(self, config: BertConfig):
        super(BackpropBertSelfAttention, self).__init__(config)
        self.query = self.convert_to_attr(self.query)
        self.key = self.convert_to_attr(self.key)
        self.value = self.convert_to_attr(self.value)

    def attr(self):
        super(BackpropBertSelfAttention, self).attr()
        self.query.attr()
        self.key.attr()
        self.value.attr()

    def attr_forward(self, hidden_states: HiddenArray,
                     attention_mask: AttentionArray = None,
                     head_mask: AttentionArray = None,
                     encoder_hidden_states: HiddenArray = None,
                     encoder_attention_mask: AttentionArray = None) -> \
            Tuple[HiddenArray, AttentionArray]:
        """
        Implements the scaled dot-product attention equation.

        :param hidden_states: The input to the attention layer
        :param attention_mask: The attention mask
        :param head_mask: An optional mask for heads
        :param encoder_hidden_states: The encoder hidden states, passed
            to this layer when used in a decoder
        :param encoder_attention_mask: None
        :return: The result of the attention equation and the attention
            probabilities
        """
        mixed_query_layer = self.query(hidden_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.hidden_to_attention(mixed_query_layer)
        key_layer = self.hidden_to_attention(mixed_key_layer)
        value_layer = self.hidden_to_attention(mixed_value_layer)

        attention_scores = query_layer @ key_layer.transpose(0, 1, 3, 2)
        attention_scores /= np.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = sp.softmax(attention_scores, axis=-1)
        if head_mask is not None:
            attention_probs *= head_mask

        context_layer = attention_probs @ value_layer
        self._state = dict(context_layer=context_layer,
                           attention_probs=attention_probs,
                           value_layer=value_layer)

        context_layer = self.attention_to_hidden(context_layer)
        return context_layer, attention_probs


class BackpropBertSelfOutput(BackpropBertMixin, bert.BertSelfOutput):
    """
    Implements the attention heads and add-and-norm portion of a self-
    attention layer. This layer is used with BackpropBertSelfAttention.
    """

    def __init__(self, config: BertConfig):
        super(BackpropBertSelfOutput, self).__init__(config)
        self.dense = self.convert_to_attr(self.dense)
        self.LayerNorm = self.convert_to_attr(self.LayerNorm)

    def attr(self):
        super(BackpropBertSelfOutput, self).attr()
        self.dense.attr()
        self.LayerNorm.attr()

    def attr_forward(self, hidden_states: HiddenArray,
                     input_tensor: HiddenArray) -> HiddenArray:
        dense_output = self.dense(hidden_states)
        self._state = dict(dense_output=dense_output,
                           input_tensor=input_tensor)
        return self.LayerNorm(dense_output + input_tensor)


class BackpropBertAttention(BackpropBertMixin, bert.BertAttention):
    """
    A complete self-attention layer, which combines
    BackpropBertSelfAttention with BackpropBertSelfOutput.
    """

    _bert_layer_types = {bert.BertSelfAttention: BackpropBertSelfAttention,
                         bert.BertSelfOutput: BackpropBertSelfOutput}

    def __init__(self, config: BertConfig):
        super(BackpropBertAttention, self).__init__(config)
        self.self = self.convert_bert_to_attr(self.self, config)
        self.output = self.convert_bert_to_attr(self.output, config)

    def attr(self):
        super(BackpropBertAttention, self).attr()
        self.self.attr()
        self.output.attr()

    def attr_forward(self, hidden_states: HiddenArray,
                     attention_mask: AttentionArray = None,
                     head_mask: AttentionArray = None,
                     encoder_hidden_states: HiddenArray = None,
                     encoder_attention_mask: AttentionArray = None) -> \
            Tuple[HiddenArray, AttentionArray]:
        """

        :param hidden_states: The attention layer input (batch_size,
            seq_len, hidden_size)
        :param attention_mask: The attention mask
        :param head_mask:
        :param encoder_hidden_states:
        :param encoder_attention_mask:
        :return: The attention layer output
        """
        self_outputs = self.self(hidden_states, attention_mask=attention_mask,
                                 head_mask=head_mask,
                                 encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask)
        return self.output(self_outputs[0], hidden_states), self_outputs[1]


class BackpropBertIntermediate(BackpropBertMixin, bert.BertIntermediate):
    """
    Implements the first linear layer after the self-attention layer.
    """

    def __init__(self, config: BertConfig):
        super(BackpropBertIntermediate, self).__init__(config)
        self.dense = self.convert_to_attr(self.dense)
        self.intermediate_act_fn_numpy = activations[config.hidden_act]

    def attr(self):
        super(BackpropBertIntermediate, self).attr()
        self.dense.attr()

    def attr_forward(self, hidden_states: HiddenArray) -> HiddenArray:
        return self.intermediate_act_fn_numpy(self.dense(hidden_states))


class BackpropBertOutput(BackpropBertMixin, bert.BertOutput):
    """
    Implements the final linear and add-and-norm layers of a Transformer
    encoder/decoder block.
    """

    def __init__(self, config: BertConfig):
        super(BackpropBertOutput, self).__init__(config)
        self.dense = self.convert_to_attr(self.dense)
        self.LayerNorm = self.convert_to_attr(self.LayerNorm)

    def attr(self):
        super(BackpropBertOutput, self).attr()
        self.dense.attr()
        self.LayerNorm.attr()

    def attr_forward(self, hidden_states: HiddenArray,
                     input_tensor: HiddenArray) -> HiddenArray:
        dense_output = self.dense(hidden_states)
        self._state = dict(dense_output=dense_output,
                           input_tensor=input_tensor)
        return self.LayerNorm(dense_output + input_tensor)


class BackpropBertLayer(BackpropBertMixin, bert.BertLayer):
    """
    A full BERT encoder or decoder block.
    """
    _bert_layer_types = {bert.BertAttention: BackpropBertAttention,
                         bert.BertIntermediate: BackpropBertIntermediate,
                         bert.BertOutput: BackpropBertOutput}

    def __init__(self, config: BertConfig):
        super(BackpropBertLayer, self).__init__(config)
        self.attention = self.convert_bert_to_attr(self.attention, config)
        if self.add_cross_attention:
            self.crossattention = self.convert_bert_to_attr(
                self.crossattention, config)
        self.intermediate = self.convert_bert_to_attr(self.intermediate,
                                                      config)
        self.output = self.convert_bert_to_attr(self.output, config)

    def attr(self):
        super(BackpropBertLayer, self).attr()
        self.attention.attr()
        if self.add_cross_attention:
            self.crossattention.attr()
        self.intermediate.attr()
        self.output.attr()

    def attr_forward(self, hidden_states: HiddenArray,
                     attention_mask: AttentionArray = None,
                     head_mask: AttentionArray = None,
                     encoder_hidden_states: HiddenArray = None,
                     encoder_attention_mask: AttentionArray = None) -> \
            Tuple[HiddenArray, AttentionArray]:
        """
        The complete forward pass for a full encoder or decoder block.

        :param hidden_states: The input to the encoder or decoder block
        :param attention_mask: The attention mask
        :param head_mask: The head mask
        :param encoder_hidden_states: Hidden states from the encoder, if
            this is a decoder block
        :param encoder_attention_mask: The encoder attention mask, if
            this is a decoder block
        :return: The output of this block, along with the attention
            scores
        """
        assert self.attention.attr_mode
        assert self.intermediate.attr_mode
        assert self.output.attr_mode
        if self.add_cross_attention:
            assert self.crossattention.attr_mode

        attn_output, attn_probs = self.attention(hidden_states,
                                                 attention_mask=attention_mask,
                                                 head_mask=head_mask)

        if self.is_decoder and encoder_hidden_states is not None:
            self._state = {"crossattention_used": True}
            assert hasattr(self, "crossattention")
            assert self.crossattention.attr_mode
            cross_output = self.crossattention(attn_output,
                                               attention_mask, head_mask,
                                               encoder_hidden_states,
                                               encoder_attention_mask)
            attn_output, attn_probs = cross_output
        else:
            self._state = {"crossattention_used": False}

        # TODO: Make apply_chunking_to_forward compatible with NumPy
        output = bert.apply_chunking_to_forward(self.feed_forward_chunk,
                                                self.chunk_size_feed_forward,
                                                self.seq_len_dim, attn_output)

        return output, attn_probs


class BackpropBertEncoder(BackpropBertMixin, bert.BertEncoder):
    """
    A BERT encoder, consisting of multiple encoder blocks.
    """

    _bert_layer_types = {bert.BertLayer: BackpropBertLayer}

    def __init__(self, config: BertConfig):
        super(BackpropBertEncoder, self).__init__(config)
        layers = [self.convert_bert_to_attr(e, config) for e in self.layer]
        self.layer = nn.ModuleList(layers)

    def attr(self):
        super(BackpropBertEncoder, self).attr()
        for e in self.layer:
            e.attr()

    def attr_forward(self, hidden_states: HiddenArray,
                     attention_mask: AttentionArray = None,
                     head_mask: AttentionArray = None,
                     encoder_hidden_states: HiddenArray = None,
                     encoder_attention_mask: AttentionArray = None) -> \
            Tuple[HiddenArray, List[HiddenArray], List[AttentionArray]]:
        """
        A full BERT encoder, consisting of multiple encoder blocks.

        :param hidden_states: The combined word, position, and token
            type embeddings
        :param attention_mask: The attention mask
        :param head_mask: The head mask
        :param encoder_hidden_states: ???
        :param encoder_attention_mask: ???
        :return: The output of the last layer, along with the outputs
            and attention scores of all layers
        """
        all_hidden_states = []
        all_attentions = []

        for i, e in enumerate(self.layer):
            all_hidden_states.append(hidden_states)

            # TODO: Add gradient checkpointing
            layer_outputs = e(hidden_states, attention_mask=attention_mask,
                              head_mask=head_mask[i],
                              encoder_hidden_states=encoder_hidden_states)

            hidden_states = layer_outputs[0]
            all_attentions.append(layer_outputs[1])

        return hidden_states, all_hidden_states, all_attentions


class BackpropBertPooler(BackpropBertMixin, bert.BertPooler):
    """
    A layer that "pools" the BERT output by passing the CLS output
    through a tanh.
    """

    def __init__(self, config: BertConfig):
        super(BackpropBertPooler, self).__init__(config)
        self.dense = self.convert_to_attr(self.dense)

    def attr(self):
        super(BackpropBertPooler, self).attr()
        self.dense.attr()

    def attr_forward(self, hidden_states: HiddenArray) -> np.ndarray:
        return np.tanh(self.dense(hidden_states[:, 0]))


class BackpropBertModel(BackpropBertMixin, bert.BertModel):
    """
    A full BERT model. This is a stack of Transformer encoders that
    takes an input sequence of the form
        [CLS] sequence1 [SEP] sequence2
    and produces an output sequence of the same form. It is pre-trained
    on BERT's masked language modeling objective.
    """
    _bert_layer_types = {bert.BertEmbeddings: BackpropBertEmbeddings,
                         bert.BertEncoder: BackpropBertEncoder,
                         bert.BertPooler: BackpropBertPooler}

    _convert_attr_input_to_numpy = False

    def __init__(self, config: BertConfig):
        super(BackpropBertModel, self).__init__(config)
        self.embeddings = self.convert_bert_to_attr(self.embeddings, config)
        self.encoder = self.convert_bert_to_attr(self.encoder, config)
        self.pooler = self.convert_bert_to_attr(self.pooler, config)

    def attr(self):
        super(BackpropBertModel, self).attr()
        self.embeddings.attr()
        self.encoder.attr()
        self.pooler.attr()

    def attr_forward(self, input_ids=None, attention_mask=None,
                     token_type_ids=None, position_ids=None, head_mask=None,
                     inputs_embeds=None, encoder_hidden_states=None,
                     encoder_attention_mask=None):
        """
        The complete BERT forward pass.

        :param input_ids: An input sequence, represented as an index
            tensor of shape (batch_size, seq_len)
        :param attention_mask: An attention mask that masks out [PAD]
            symbols and symbols without a prediction
        :param token_type_ids: Not sure what this is for
        :param position_ids: The positional encoding
        :param head_mask: Some other mask
        :param inputs_embeds: Embedding vectors for the input. This
            cannot be specified if input_ids is specified, and vice
            versa
        :param encoder_hidden_states: Hidden states from a previous
            computation, which will be reused
        :param encoder_attention_mask: The attention mask from a
            previous computation, which will be reused

        :return: The sequence output, the pooled output, and all the
            encoder block outputs
        """
        # Get input embedding shape
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and "
                             "inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or "
                             "inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = np.zeros(input_shape, dtype="int64")

        # Not really sure what this is for
        extended_attention_mask = \
            self.get_extended_attention_mask(attention_mask, input_shape,
                                             torch.device("cpu"))
        extended_attention_mask = extended_attention_mask.detach().numpy()

        if self.config.is_decoder and encoder_hidden_states is not None:
            raise RuntimeWarning("I didn't implement this carefully")
            encoder_hidden_shape = encoder_hidden_states.shape[:-1]
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape)
            encoder_extended_attn_mask = \
                self.invert_attention_mask(encoder_attention_mask)
            encoder_extended_attn_mask = \
                encoder_extended_attn_mask.detach().numpy()
        else:
            encoder_extended_attn_mask = None

        head_mask = self.get_head_mask(head_mask,
                                       self.config.num_hidden_layers)

        # Begin forward pass
        embedding_output = self.embeddings(input_ids=input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids,
                                           inputs_embeds=inputs_embeds)

        encoder_outputs = \
            self.encoder(embedding_output,
                         attention_mask=extended_attention_mask,
                         head_mask=head_mask,
                         encoder_hidden_states=encoder_hidden_states,
                         encoder_attention_mask=encoder_extended_attn_mask)

        sequence_output = encoder_outputs[0]
        self._state = {"output_shape": sequence_output.shape}
        pooled_output = self.pooler(sequence_output)
        return (sequence_output, pooled_output) + encoder_outputs[1:]


BFSC = bert.BertForSequenceClassification


class BackpropBertForSequenceClassification(BackpropBertMixin, BFSC):
    """
    A BERT model with a linear decoder.
    """
    _bert_layer_types = {bert.BertModel: BackpropBertModel}

    _convert_attr_input_to_numpy = False

    def __init__(self, config: BertConfig):
        super(BackpropBertForSequenceClassification, self).__init__(config)
        self.bert = self.convert_bert_to_attr(self.bert, config)
        self.classifier = self.convert_to_attr(self.classifier)

    def attr(self):
        super(BackpropBertForSequenceClassification, self).attr()
        self.bert.attr()
        self.classifier.attr()

    def attr_forward(self, **kwargs):
        outputs = self.bert(**kwargs)
        return self.classifier(outputs[1])
