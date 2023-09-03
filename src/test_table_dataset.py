from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("e2e_nlg")
print(dataset.keys())

for i in range(20):
    print(dataset["train"][i])


tokenizer = AutoTokenizer.from_pretrained("t5-small")

for data_set_name in dataset.keys():
    print(data_set_name)
    data = dataset[data_set_name]
    attrs = data[0].keys()
    print(attrs)
    for attr in attrs:
            text = list(map(lambda x: x[attr], data))
            m = max([len(tokenizer(x, max_length=None).input_ids) for x in text])
            print(f"{attr}: {m}")
    print("\n\n")
