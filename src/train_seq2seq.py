from models import Seq2Seq
from trainer import Seq2SeqTrainer
from transformers import AutoTokenizer

config = {
    "src_max_length": 183,
    "dst_max_length": 208,
    "src_vocab_size": 31102,
    "dst_vocab_size": 28996,
    "max_epochs": 10,
    "batch_size": 64,
    "seed": 7
}

model = Seq2Seq(
    config["src_vocab_size"],
    config["dst_vocab_size"],
    256,
    1,
    1,
    1
)


src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
trainer = Seq2SeqTrainer(src_tokenizer, dst_tokenizer, config)