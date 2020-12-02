from typing import Tuple

import numpy as np
from torch import nn
from transformers.models.bert import modeling_bert as bert

from interpret_nlp.attribution.lrp import lrp_linear, lrp_matmul
from interpret_nlp.modules import backprop_bert as bpbert
from interpret_nlp.modules.backprop_bert import HiddenArray
from interpret_nlp.modules.lrp_modules import LRPLinear, LRPLayerNorm


class LRPBertMixin(bpbert.BackpropBertMixin):
    _layer_types = {nn.Linear: LRPLinear,
                    nn.LayerNorm: LRPLayerNorm}


class LRPBertEmbeddings(LRPBertMixin, bpbert.BackpropBertEmbeddings):
    """
    BertEmbeddings with LRP.
    """

    def attr_backward(self, rel_y: HiddenArray, eps: float = 0.001) -> \
            Tuple[HiddenArray, HiddenArray, HiddenArray]:
        """

        :param rel_y:
        :param eps:
        :return:
        """
        rel_y = self.LayerNorm.attr_backward(rel_y, eps=eps)

        inp_embeds, pos_embeds, tok_type_embeds = self._state
        combined_embeds = inp_embeds + pos_embeds + tok_type_embeds
        rel_input = lrp_linear(inp_embeds, combined_embeds, rel_y, eps=eps)
        rel_pos = lrp_linear(pos_embeds, combined_embeds, rel_y, eps=eps)
        rel_tok = lrp_linear(tok_type_embeds, combined_embeds, rel_y, eps=eps)
        return rel_input, rel_pos, rel_tok


class LRPBertSelfAttention(LRPBertMixin, bpbert.BackpropBertSelfAttention):
    """
    BertSelfAttention with LRP.
    """

    def attr_backward(self, rel_y: HiddenArray, eps: float = 0.001) -> \
            HiddenArray:
        """
        All relevance gets propagated to the value layer.

        :param rel_y:
        :param eps:
        :return:
        """
        rel_value_layer = lrp_matmul(self._state["value_layer"],
                                     self._state["attention_probs"],
                                     self._state["context_layer"],
                                     self.hidden_to_attention(rel_y),
                                     eps=eps)

        rel_value_layer = self.attention_to_hidden(rel_value_layer)
        return self.value.attr_backward(rel_value_layer)


class LRPBertSelfOutput(LRPBertMixin, bpbert.BackpropBertSelfOutput):
    """
    BertSelfOutput with LRP.
    """

    def attr_backward(self, rel_y: HiddenArray, eps: float = 0.001) -> \
            Tuple[HiddenArray, HiddenArray]:
        input_tensor = self._state["input_tensor"]
        dense_output = self._state["dense_output"]
        pre_layer_norm = input_tensor + dense_output

        rel_pre_layer_norm = self.LayerNorm.attr_backward(rel_y)
        rel_input_tensor = lrp_linear(input_tensor, pre_layer_norm,
                                      rel_pre_layer_norm, eps=eps)
        rel_dense_output = lrp_linear(dense_output, pre_layer_norm,
                                      rel_pre_layer_norm, eps=eps)
        rel_hidden_states = self.dense.attr_backward(rel_dense_output)

        return rel_hidden_states, rel_input_tensor


class LRPBertAttention(LRPBertMixin, bpbert.BackpropBertAttention):
    """
    BertAttention with LRP.
    """
    _bert_layer_types = {bert.BertSelfAttention: LRPBertSelfAttention,
                         bert.BertSelfOutput: LRPBertSelfOutput}

    def attr_backward(self, rel_y: HiddenArray, eps: float = 0.001) -> \
            HiddenArray:
        rel_hidden, rel_input = self.output.attr_backward(rel_y, eps=eps)
        rel_input += self.self.attr_backward(rel_hidden, eps=eps)
        return rel_input


class LRPBertIntermediate(LRPBertMixin, bpbert.BackpropBertIntermediate):
    """
    BertIntermediate with LRP.
    """

    def attr_backward(self, rel_y: HiddenArray, eps: float = 0.001) -> \
            HiddenArray:
        return self.dense.attr_backward(rel_y, eps=eps)


class LRPBertOutput(LRPBertMixin, bpbert.BackpropBertOutput):
    """
    BertOutput with LRP.
    """

    def attr_backward(self, rel_y: HiddenArray, eps: float = 0.001) -> \
            Tuple[HiddenArray, HiddenArray]:
        input_tensor = self._state["input_tensor"]
        dense_output = self._state["dense_output"]
        pre_layer_norm = input_tensor + dense_output

        rel_pre_layer_norm = self.LayerNorm.attr_backward(rel_y)
        rel_input_tensor = lrp_linear(input_tensor, pre_layer_norm,
                                      rel_pre_layer_norm, eps=eps)
        rel_dense_output = lrp_linear(dense_output, pre_layer_norm,
                                      rel_pre_layer_norm, eps=eps)
        rel_hidden_states = self.dense.attr_backward(rel_dense_output)

        return rel_hidden_states, rel_input_tensor


class LRPBertLayer(bpbert.BackpropBertLayer):
    """
    BertLayer with LRP.
    """
    _bert_layer_types = {bert.BertAttention: LRPBertAttention,
                         bert.BertIntermediate: LRPBertIntermediate,
                         bert.BertOutput: LRPBertOutput}

    def attr_backward(self, rel_y: HiddenArray, eps: float = 0.001) -> \
            HiddenArray:
        rel_intermediate, rel_attn = self.output.attr_backward(rel_y, eps=eps)
        rel_attn += self.intermediate.attr_backward(rel_intermediate, eps=eps)

        if self._state["crossattention_used"]:
            rel_attn = self.crossattention.attr_backward(rel_attn, eps=eps)
        return self.attention.attr_backward(rel_attn, eps=eps)


class LRPBertEncoder(bpbert.BackpropBertEncoder):
    """
    BertEncoder with LRP.
    """
    _bert_layer_types = {bert.BertLayer: LRPBertLayer}

    def attr_backward(self, rel_y: HiddenArray, eps: float = 0.001) -> \
            HiddenArray:
        rel = rel_y
        for e in reversed(self.layer):
            rel = e.attr_backward(rel, eps=eps)
        return rel


class LRPBertPooler(LRPBertMixin, bpbert.BackpropBertPooler):
    """
    BertPooler with LRP.
    """

    def attr_backward(self, rel_y: np.ndarray, eps: float = 0.001) -> \
            HiddenArray:
        return self.dense.attr_backward(rel_y, eps=eps)


class LRPBertModel(bpbert.BackpropBertModel):
    """
    BertModel with LRP.
    """
    _bert_layer_types = {bert.BertEmbeddings: LRPBertEmbeddings,
                         bert.BertEncoder: LRPBertEncoder,
                         bert.BertPooler: LRPBertPooler}

    def attr_backward(self, rel_sequence: HiddenArray = None,
                      rel_pooled: np.ndarray = None, eps: float = 0.001) -> \
            Tuple[HiddenArray, HiddenArray, HiddenArray]:
        assert rel_sequence is not None or rel_pooled is not None

        if rel_sequence is None:
            rel_sequence = np.zeros(self._state["output_shape"])
        if rel_pooled is not None:
            rel_first = self.pooler.attr_backward(rel_pooled, eps=eps)
            rel_sequence[:, 0] += rel_first

        rel_embeddings = self.encoder.attr_backward(rel_sequence, eps=eps)
        return self.embeddings.attr_backward(rel_embeddings) + \
               (rel_embeddings,)


BBFSC = bpbert.BackpropBertForSequenceClassification


class LRPBertForSequenceClassification(LRPBertMixin, BBFSC):
    """
    A BERT model with a linear decoder.
    """
    _bert_layer_types = {bert.BertModel: LRPBertModel}

    def attr_backward(self, rel_y: np.ndarray, eps: float = 0.001) -> \
            Tuple[HiddenArray, HiddenArray, HiddenArray]:
        rel_pooled = self.classifier.attr_backward(rel_y, eps=eps)
        return self.bert.attr_backward(rel_pooled=rel_pooled, eps=eps)
