from trainer import BartForNMT
from trainer import AutoTokenizer
from trainer import Trainer

# model = BartModel.from_pretrained('facebook/bart-base')

config = {
    "src_max_length": 183,
    "dst_max_length": 208,
    "src_vocab_size": 31102,
    "dst_vocab_size": 28996,
    "max_epochs": 10,
    "batch_size": 64,
    "seed": 7
}


model = BartForNMT(768, 50265)
src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
trainer = Trainer(model, src_tokenizer, dst_tokenizer, config)
trainer.train()