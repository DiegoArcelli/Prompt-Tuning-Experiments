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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = evaluate.load("accuracy")
dataset = load_dataset("super_glue", 'rte')

model, tokenizer = load_model(mode="prompt", model_type="sequence_classification", model_name="bert-base-uncased")

def tokenize_function(examples):
    outputs = tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=None)
    return outputs

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["premise", "hypothesis", "idx"]
)

train_data = tokenized_datasets["train"]
valid_data = tokenized_datasets["validation"]
test_data = tokenized_datasets["test"]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="output/",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=4,
    num_train_epochs=4,
    fp16=True,
    push_to_hub=False,
    logging_strategy="steps",
    logging_steps=100,
    logging_dir="logs/",
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
    load_best_model_at_end=True,
)

trainer.train()

train_results = trainer.evaluate(train_data)
valid_results = trainer.evaluate(valid_data)
test_results = trainer.evaluate(test_data)

print(train_results)
print(valid_results)
print(test_results)