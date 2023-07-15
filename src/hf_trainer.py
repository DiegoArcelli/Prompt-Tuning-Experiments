import sys
sys.path.append("./../")
sys.path.append("./../trainer/")
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from models.nmt_models import T5ForNMT

config = {
    "src_max_length": 126,
    "dst_max_length": 109,
    "src_vocab_size": 32100,
    "dst_vocab_size": 31102,
    "enc_hidden_dim": 64,
    "dec_hidden_dim": 64,
    "max_epochs": 1,
    "batch_size": 32,
    "seed": 7,
    "device": "cpu",
    "lang": "ita",
    "prefix": True
}

src_tokenizer = AutoTokenizer.from_pretrained("t5-small")
dst_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")

model = T5ForNMT.from_pretrained("t5-small", hidden_size=512, voc_size=config["dst_vocab_size"])

# trainer = Seq2SeqTrainer(
#     model=model,
#     # args=training_args,
#     # train_dataset=train_dataset if training_args.do_train else None,
#     # eval_dataset=eval_dataset if training_args.do_eval else None,
#     # tokenizer=tokenizer,
#     # data_collator=data_collator,
#     # compute_metrics=compute_metrics if training_args.predict_with_generate else None,
# )
