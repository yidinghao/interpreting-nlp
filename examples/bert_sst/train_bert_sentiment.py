"""
This script fine-tunes a BERT model on the SST.
"""
import torch
from torch import optim
from torchtext import data as tt
from torchtext.datasets import SST
from transformers import BertTokenizer, BertForSequenceClassification

# Load a pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      return_dict=True,
                                                      num_labels=4)
model.to("cuda")
optimizer = optim.Adam(model.parameters(), lr=3e-5, eps=1e-8)

# Load the data
text_field = tt.RawField()
label_field = tt.Field(sequential=False)
data = SST.splits(text_field, label_field)
label_field.build_vocab(data[0])
iters = tt.BucketIterator.splits(data, batch_size=32, device="cuda")

# Train
best_accuracy = 0
for epoch in range(3):
    print("Epoch", epoch + 1)
    model.train()
    for i, batch in enumerate(iters[0]):
        model.zero_grad()

        # Forward pass
        inputs = tokenizer(batch.text, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")

        output = model(**inputs, labels=batch.label)

        # Compute accuracy
        if True:
            predictions = output.logits.argmax(-1)
            num_correct = int(sum(predictions == batch.label))
            accuracy = num_correct / len(batch.label) * 100
            print("Batch {}: {}/{} correct ({:.1f}%); loss = {}".format(
                i + 1, num_correct, len(batch.label), accuracy,
                float(output.loss)))

        # Backward pass
        output.loss.backward()
        optimizer.step()

    # Dev accuracy
    model.eval()
    model.zero_grad()
    num_correct = 0
    num_total = 0
    for batch in iters[1]:
        # Forward pass
        inputs = tokenizer(batch.text, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")

        output = model(**inputs, labels=batch.label)

        # Compute accuracy
        predictions = output.logits.argmax(-1)
        num_correct += int(sum(predictions == batch.label))
        num_total += len(batch.label)

    accuracy = num_correct / num_total * 100
    print("Epoch {}: {}/{} correct ({:.1f}%)".format(
        epoch + 1, num_correct, num_total, accuracy))

    if accuracy > best_accuracy:
        torch.save(model.state_dict(), "bert-sst.pt")
        best_accuracy = accuracy

# Testing
num_correct = 0
num_total = 0
for batch in iters[2]:
    # Forward pass
    inputs = tokenizer(batch.text, return_tensors="pt", padding=True)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to("cuda")

    output = model(**inputs, labels=batch.label)

    # Compute accuracy
    predictions = output.logits.argmax(-1)
    num_correct += int(sum(predictions == batch.label))
    num_total += len(batch.label)

accuracy = num_correct / num_total * 100
print("Test: {}/{} correct ({:.1f}%)".format(num_correct, num_total, accuracy))

torch.save(model.config, "bert-sst-config.pt")
