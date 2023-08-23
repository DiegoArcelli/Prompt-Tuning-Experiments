from glue_config import glue_config
from datasets import load_dataset


for task in glue_config.keys():
    print(task)
    print(glue_config[task])
    dataset = load_dataset("glue", task)
    print(dataset)

    print("\n\n")   