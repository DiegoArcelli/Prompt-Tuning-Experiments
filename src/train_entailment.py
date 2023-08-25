from transformers import T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, T5Tokenizer
import evaluate
import numpy as np
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import random_split
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from seq2seq_trainer_prompt import Seq2SeqTrainerPrompt
from utils import load_model

num_text_map = {
    0: "entailment",
    1: "not entailment",
    -1: "don't know"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = load_model(mode="normal", model_type="generation", model_name="t5-small")

metric = evaluate.load("accuracy")
dataset = load_dataset("super_glue", 'rte')

text_attr1 = "premise"
text_attr2 = "hypothesis"

def tokenize_dataset(dataset):

    records = []
    with tqdm(total=len(dataset)) as pbar:
        for idx, data in enumerate(dataset):
            record = {}
            premise = data[text_attr1]
            hypothesis = data[text_attr2]
            text = f"rte sentence 1: {premise} sentence 2: {hypothesis}"
            label = data["label"]
            label = num_text_map[label]

            tokenized_text = tokenizer(text, max_length=None, truncation=True) 
            tokenized_label = tokenizer(label, max_length=6, truncation=True)

            record["id"] = idx
            record["input_ids"] = tokenized_text.input_ids
            record["attention_mask"] = tokenized_text.attention_mask
            record["labels"] = tokenized_label.input_ids
            records.append(record)

            pbar.update(1)

        return records
    

test_split = 0.1
test_size = int(len(dataset["train"])*test_split)
train_test_set = dataset["train"].train_test_split(test_size)

train_data = Dataset.from_list(tokenize_dataset(train_test_set["train"]))
test_data = Dataset.from_list(tokenize_dataset(train_test_set["test"]))
valid_data = Dataset.from_list(tokenize_dataset(dataset["validation"]))

print(train_data)
print(test_data)
print(valid_data)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print(preds, labels)

    correct = 0
    total = 0
    for pred, true in zip(preds, labels):
        if pred.strip() == true.strip():
            correct += 1
        total += 1
    accuracy = correct / total
    return {"accuracy": accuracy}


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")

training_args = Seq2SeqTrainingArguments(
    output_dir="output/",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    logging_strategy="steps",
    logging_steps=10,
    logging_dir="logs/",
    #disable_tqdm=True
)

trainer = Seq2SeqTrainerPrompt(
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
