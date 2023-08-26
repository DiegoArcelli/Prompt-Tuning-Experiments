import sys
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, T5Tokenizer, AutoModelForAudioClassification
import evaluate
import numpy as np
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import random_split
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from seq2seq_trainer_prompt import Seq2SeqTrainerPrompt
from utils import load_model
from glue_config import glue_config
import argparse


parser = argparse.ArgumentParser( prog='Train GLUE', description='Train GLUE')
parser.add_argument('-t', '--task', default="mrpc", type=str)
parser.add_argument('-m', '--mode', default="normal", type=str)    
parser.add_argument('-lr', '--learning_rate', default=3e-5, type=float)
parser.add_argument('-e', '--epochs', default=5, type=int)
parser.add_argument('-b', '--batch_size', default=4, type=int)

args = parser.parse_args()

task = args.task
mode = args.mode
lr = args.learning_rate
num_epochs = args.epochs
batch_size = args.batch_size

task_config = glue_config[task]

num_attrs = len(task_config.keys())
attr1 = task_config["attribute_1"]
attr2 = None if num_attrs == 1 else task_config["attribute_2"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("glue", task)
metric = evaluate.load("glue", task)

model, tokenizer = load_model(mode=mode, model_type="sequence_classification", model_name="bert-base-uncased")

def tokenize_function(examples):
    if num_attrs == 2:
        outputs = tokenizer(examples[attr1], examples[attr2], truncation=True, max_length=None)
    else:
        outputs = tokenizer(examples[attr1], truncation=True, max_length=None)
    return outputs

def tokenize_dataset(dataset):

    records = []
    with tqdm(total=len(dataset)) as pbar:
        for idx, data in enumerate(dataset):
            record = {}
            premise = data[attr1]
            if num_attrs == 2:
                tokenized_text = tokenizer(data[attr1], data[attr2], truncation=True, max_length=None)
            else:
                tokenized_text = tokenizer(data[attr1], truncation=True, max_length=None)
            

            record["id"] = idx
            record["input_ids"] = tokenized_text.input_ids
            record["attention_mask"] = tokenized_text.attention_mask
            record["labels"] = data["label"]
            records.append(record)

            pbar.update(1)

        return records

# tokenized_datasets = dataset.map(
#     tokenize_function,
#     batched=True,
#     batch_size=1,
#     remove_columns=[attr1, attr2, "idx"] if attr2 is not None else [attr1, "idx"],
# )

# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

valid_data = tokenize_dataset(dataset["validation"])
test_split = 0.1
test_size = int(len(dataset["train"])*test_split)
train_test_set = dataset["train"].train_test_split(test_size)
train_data = tokenize_dataset(train_test_set["train"])
test_data = tokenize_dataset(train_test_set["test"])

train_data = Dataset.from_list(train_data)
valid_data = Dataset.from_list(valid_data)
test_data = Dataset.from_list(test_data)

print(train_data)
print(valid_data)
print(test_data)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="output/",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=4,
    num_train_epochs=num_epochs,
    lr_scheduler_type="linear",
    adam_beta1=0.9,
    adam_beta2=0.99,
    adam_epsilon=1e-8,
    fp16=True,
    push_to_hub=False,
    logging_strategy="steps",
    logging_steps=100,
    logging_dir="logs/",
    load_best_model_at_end=True,
    #disable_tqdm=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

train_results = trainer.evaluate(train_data)
valid_results = trainer.evaluate(valid_data)
test_results = trainer.evaluate(test_data)

print(train_results)
print(valid_results)
print(test_results)