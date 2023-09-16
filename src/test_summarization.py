from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("billsum")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# print(dataset["train"][0]["text"])
# print("=================================")
# print(dataset["train"][0]["summary"])
# print("=================================")
# print(dataset["train"][0]["title"])

print(len(dataset["train"]))
print(len(dataset["test"]))

for data_set_name in dataset.keys():
    print(data_set_name)
    data = dataset[data_set_name]
    attrs = data[0].keys()
    print(attrs)
    for attr in attrs:
        if attr != "idx" and attr != "label":
            text = list(map(lambda x: x[attr], data))
            m = max([len(tokenizer(x, max_length=None).input_ids) for x in text])
            print(f"{attr}: {m}")
    print("\n\n")
