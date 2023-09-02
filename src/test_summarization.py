from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("glue", "stsb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


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
