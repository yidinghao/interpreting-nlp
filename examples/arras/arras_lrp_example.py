"""
This script replicates Leila Arras's example script:
https://github.com/ArrasL/LRP_for_LSTM/blob/master/run_example.ipynb
"""
import os
import pickle

import numpy as np
import requests
import torch
from torch import nn

from modules.lrp_modules import LRPLSTM, LRPLinear


def download(filename: str):
    if os.path.isfile(filename):
        return

    url = "https://github.com/ArrasL/LRP_for_LSTM/raw/master/model/" + filename
    downloaded_file = requests.get(url)
    with open(filename, "wb") as f:
        f.write(downloaded_file.content)


def convert_lstm_weight(old):
    w_i, w_g, w_f, w_o = np.split(old, 4)
    return torch.tensor(np.concatenate((w_i, w_f, w_g, w_o)))


def convert_state_dict(old, embeddings):
    l = "_Left"
    r = "_Right"
    lin_weight = np.concatenate((old["Why" + l], old["Why" + r]), axis=1)
    new_state_dict = {"lstm.weight_ih_l0": convert_lstm_weight(old["Wxh" + l]),
                      "lstm.weight_hh_l0": convert_lstm_weight(old["Whh" + l]),
                      "lstm.bias_ih_l0": convert_lstm_weight(old["bxh" + l]),
                      "lstm.bias_hh_l0": convert_lstm_weight(old["bhh" + l]),
                      "lstm.weight_ih_l0_reverse":
                          convert_lstm_weight(old["Wxh" + r]),
                      "lstm.weight_hh_l0_reverse":
                          convert_lstm_weight(old["Whh" + r]),
                      "lstm.bias_ih_l0_reverse":
                          convert_lstm_weight(old["bxh" + r]),
                      "lstm.bias_hh_l0_reverse":
                          convert_lstm_weight(old["bhh" + r]),
                      "linear.weight": torch.tensor(lin_weight),
                      "linear.bias": torch.zeros(5),
                      "embedding.weight": torch.tensor(embeddings)}

    return new_state_dict


class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.lstm = LRPLSTM(60, 60, bidirectional=True)
        self.embedding = nn.Embedding(19538, 60)
        self.linear = LRPLinear(120, 5)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        embeddings = self.embedding(x)
        hidden = self.lstm(embeddings)[1][0].view(1, -1)
        return self.linear(hidden)


if __name__ == "__main__":
    # Download files from Arras's GitHub repository
    download("model")
    download("vocab")
    download("embeddings.npy")

    # Load model
    with open("model", "rb") as f:
        arras_state_dict = pickle.load(f)
    vectors = np.load("embeddings.npy")
    state_dict = convert_state_dict(arras_state_dict, vectors)

    model = LSTMClassifier()
    model.load_state_dict(state_dict)
    model.eval()

    # Load vocab
    with open("vocab", "rb") as f:
        vocab = {w: i for i, w in enumerate(pickle.load(f))}

    # Prepare input
    sentence = "neither funny nor suspenseful nor particularly well-drawn ."
    sentence = sentence.split()
    x = torch.LongTensor([[vocab[w] for w in sentence]])
    y = model(x).detach().numpy()
    print("PyTorch Output:", y, sep="\n    ")

    # LRP forward
    model.linear.attr()
    model.lstm.attr()

    embeddings = model.embedding(x)
    lstm_out = model.lstm(embeddings.detach().numpy())  # 1, 8, 120
    h_final = np.concatenate((lstm_out[:, -1, :60], lstm_out[:, 0, 60:]),
                             axis=-1)
    output = model.linear(h_final)
    print("NumPy Output:", output, sep="\n    ")

    # LRP backward
    target_class = np.argmax(output, 1)
    rel_output = np.zeros(output.shape)
    rel_output[:, target_class] = output[:, target_class]
    rel_h_final = model.linear.attr_backward(rel_output)

    rel_lstm_out = np.zeros(lstm_out.shape)
    rel_lstm_out[:, -1, :60] = rel_h_final[:, :60]
    rel_lstm_out[:, 0, 60:] = rel_h_final[:, 60:]
    rel_embeddings = model.lstm.attr_backward(rel_lstm_out)

    print("Relevance Scores:")
    for i, rel_w in enumerate(rel_embeddings.sum(axis=-1).squeeze()):
        print("    {}: {:.4f}".format(sentence[i], rel_w))
