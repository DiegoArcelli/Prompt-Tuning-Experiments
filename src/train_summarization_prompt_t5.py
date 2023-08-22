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
from models.prompt_tuning_models import T5PromptTuningSimple


def load_model(prompt_tuning=False):
    if prompt_tuning == False:
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
    else:
        model = T5PromptTuningSimple.from_pretrained(
            "t5-small",
            encoder_soft_prompt_path = None,
            decoder_soft_prompt_path = None,
            encoder_n_tokens = 20,
            decoder_n_tokens = 20,
            # encoder_hidden_dim = 512,
            # decoder_hidden_dim = 512,
            device=device
        )
    return model


tokenizer = T5Tokenizer.from_pretrained("t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


metric = evaluate.load("rouge")
dataset = load_dataset("billsum")


def tokenize_dataset(data):

    records = []
    with tqdm(total=len(data)) as pbar:
        for idx, data in enumerate(data):
            record = {}
            tokenized_text = tokenizer(data["text"], max_length=1024, truncation=True, return_tensors='pt')
            tokenized_summary = tokenizer(data["summary"], max_length=128, truncation=True, return_tensors='pt')

            for key in tokenized_text.keys():
                tokenized_text[key] = tokenized_text[key][0]
                tokenized_summary[key] = tokenized_summary[key][0]

            record["id"] = idx

            record["input_ids"] = tokenized_text.input_ids
            record["attention_mask"] = tokenized_text.attention_mask
            record["labels"] = tokenized_summary.input_ids
            records.append(record)

            if idx == 100:
                break

            pbar.update(1)

        return records


train_data = tokenize_dataset(dataset["train"])
test_data = tokenize_dataset(dataset["test"])

val_split = 0.1
val_size = int(len(train_data)*val_split)

train_data = Dataset.from_list(train_data)
test_data = Dataset.from_list(test_data)
train_val_data = train_data.train_test_split(test_size=val_size)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


model = load_model(prompt_tuning=False)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")

training_args = Seq2SeqTrainingArguments(
    output_dir="output/",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    logging_strategy="steps",
    logging_steps=100,
    logging_dir="logs/",
    #disable_tqdm=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_val_data["train"],
    eval_dataset=train_val_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

train_results = trainer.evaluate(train_val_data["train"])
valid_results = trainer.evaluate(train_val_data["test"])
test_results = trainer.evaluate(test_data)

print(train_results)
print(valid_results)
print(test_results)