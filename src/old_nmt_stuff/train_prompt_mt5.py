import sys
sys.path.append("./models/")
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from nmt_datasets import AnkiDatasetFactory
from transformers import AutoTokenizer
import evaluate
import numpy as np
from transformers import DataCollatorForSeq2Seq
from models.prompt_tuning_models_multiple import MT5PromptTuningSimple
from transformers.utils import logging
from torch.nn import Linear
import torch


# logging.set_verbosity_info()
# logger = logging.get_logger("transformers")
# logger.info("INF(data_set.test_data["test"])
# logger.warning("WARN")

config = {
    "src_max_length": 183,
    "dst_max_length": 208,
    "src_vocab_size": 32128,
    "dst_vocab_size": 28996,
    "enc_hidden_dim": 8,
    "dec_hidden_dim": 8,
    "max_epochs": 1,
    "batch_size": 16,
    "seed": 7,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = dst_tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, dst_tokenizer.pad_token_id)
    decoded_labels = dst_tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != dst_tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


src_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
dst_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")


# model = T5ForNMT.from_pretrained("t5-small", hidden_size=512, voc_size=config["dst_vocab_size"])
model = MT5PromptTuningSimple.from_pretrained(
    "google/mt5-small",
    encoder_soft_prompt_path = None,
    decoder_soft_prompt_path = None,
    encoder_n_tokens = 20,
    decoder_n_tokens = 20,
    device=device
)

model = model.to(device)

data_collator = DataCollatorForSeq2Seq(tokenizer=dst_tokenizer, model="google/mt5-small")
# model.lm_head = Linear(in_features=512, out_features=31102, bias=False)


data_set = AnkiDatasetFactory(
            f"../dataset/ita.txt",
            src_tokenizer,
            dst_tokenizer,
            config["src_max_length"],
            config["dst_max_length"],
            subsample=True,
            frac=0.001,
            seed=7,
            lang="ita"
        )

training_args = Seq2SeqTrainingArguments(
    output_dir="output/",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=0.15,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
    logging_strategy="steps",
    logging_steps=100,
    logging_dir="logs/",
    #disable_tqdm=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=data_set.train_val_data["train"],
    eval_dataset=data_set.train_val_data["test"],
    tokenizer=dst_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

train_results = trainer.evaluate(data_set.train_val_data["train"])
valid_results = trainer.evaluate(data_set.train_val_data["test"])
test_results = trainer.evaluate(data_set.test_data["test"])

print(train_results)
print(valid_results)
print(test_results)
