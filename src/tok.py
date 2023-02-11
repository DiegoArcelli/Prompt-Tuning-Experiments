from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("rotten_tomatoes", split="train")
# dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
# dataset.format['type']

def tokenize(example):
    return tokenizer(example["text"])

dataset = dataset.map(tokenize, batched=True)

print(len(dataset))

for data in dataset:
    print(len(data["input_ids"]))