import numpy as np
import torch
from transformers import BertTokenizer

from modules.lrp_bert_modules import LRPBertForSequenceClassification

# Load the model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = LRPBertForSequenceClassification(torch.load("bert-sst-config.pt"))
model.load_state_dict(torch.load("bert-sst.pt"))
model.eval()

# Test
test_example = "It's a lovely film with lovely performances by Buy and " \
               "Accorsi."
print("Example Input:", test_example, end="\n\n")

inputs = tokenizer(test_example, return_tensors="pt")
logits = model(**inputs).logits.squeeze()
classes = ["<unk>", "positive", "negative", "neutral"]
print("Logit Scores:")
for c, score in zip(classes, logits):
    print("{}: {}".format(c, score))

# Test attr forward
model.attr()
print("\nAttr Forward Pass Output:")
output = model(**inputs)
print(output)

# Test LRP
tokens = tokenizer.tokenize(test_example)
rel_y = np.zeros(output.shape)
rel_y[:, 1] = output[:, 1]
rel_embeddings, _, _ = model.attr_backward(rel_y)
scores = np.sum(rel_embeddings[0, :, 1:-2], -1)

print("\nLRP Scores:")
for t, s in zip(tokens, scores):
    print(t, s, sep=": ")
