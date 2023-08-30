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
num_text_map = task_config["num_to_text"]
text_num_map = {v: k for (k, v) in num_text_map.keys()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("glue", task)
metric = evaluate.load("glue", task)

model, tokenizer = load_model(mode=mode, model_type="text_generation", model_name="t5-small")




def tokenize_dataset(dataset):

    records = []
    with tqdm(total=len(dataset)) as pbar:
        for idx, data in enumerate(dataset):
            record = {}
            premise = data[attr1]
            hypothesis = data[attr2]

            if num_attrs != 2:
                text = f"{task} {data[attr1]}"
            else:
                text = f"{task} "
            label = data["label"]
            label = num_text_map[label]

            tokenized_text = tokenizer(text, max_length=None, truncation=True) 
            tokenized_label = tokenizer(label, max_length=None, truncation=True)

            record["id"] = idx
            record["input_ids"] = tokenized_text.input_ids
            record["attention_mask"] = tokenized_text.attention_mask
            record["labels"] = tokenized_label.input_ids
            records.append(record)

            pbar.update(1)

        return records
    
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=preds, references=labels)
    # print(preds, labels)

    # correct = 0
    # total = 0
    # for pred, true in zip(preds, labels):
    #     if pred.strip() == true.strip():
    #         correct += 1
    #     total += 1
    # accuracy = correct / total
    # return {"accuracy": accuracy}


valid_data = tokenize_dataset(dataset["validation"])
test_split = 0.2
test_size = int(len(dataset["train"])*test_split)
train_test_set = dataset["train"].train_test_split(test_size, stratify_by_column="label", seed=42)
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
    predict_with_generate=True,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=num_epochs,
    lr_scheduler_type="linear",
    optim="adamw_hf",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    fp16=True,
    push_to_hub=False,
    logging_strategy="steps",
    logging_steps=100,
    logging_dir="logs/",
    load_best_model_at_end=True,
    seed=42
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