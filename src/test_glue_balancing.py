from datasets import load_dataset
import numpy as np

for task in ["cola", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"]:
    print(task)
    dataset = load_dataset("glue", task)
    print(dataset.keys())
    train_set = dataset["train"]
    valid_set = dataset["validation"]
    train_labels = list(map(lambda x: x["label"], train_set))
    valid_labels = list(map(lambda x: x["label"], valid_set))
    print(set(train_labels), set(valid_labels))
    tot_train = len(train_labels)
    tot_valid = len(valid_labels)
    class_1_train = sum(train_labels)
    class_0_train = tot_train - class_1_train
    class_1_valid = sum(valid_labels)
    class_0_valid = tot_valid - class_1_valid
    print(f"Train class 0: {class_0_train} ({class_0_train/tot_train}), class 1: {class_1_train} ({class_1_train/tot_train})")
    print(f"Valid class 0: {class_0_valid} ({class_0_valid/tot_valid}), class 1: {class_1_valid} ({class_1_valid/tot_valid})\n\n")

