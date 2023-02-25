from models.rnn_models import Seq2Seq
from models.nmt_models import BartForNMT, T5ForNMT
from trainers.seq2seq_trainer import Seq2SeqTrainer
from transformers import AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "src_max_length": 183,
    "dst_max_length": 208,
    "src_vocab_size": 31102,
    "dst_vocab_size": 28996,
    "enc_hidden_dim": 8,
    "dec_hidden_dim": 8,
    "max_epochs": 3,
    "batch_size": 8,
    "seed": 7
}

model = Seq2Seq(
    config["src_vocab_size"],
    config["dst_vocab_size"],
    config["enc_hidden_dim"],
    config["dec_hidden_dim"],
    1,
    1,
    0.5,
    device
)

src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
trainer = Seq2SeqTrainer(model, src_tokenizer, dst_tokenizer, config)

trainer.train()
