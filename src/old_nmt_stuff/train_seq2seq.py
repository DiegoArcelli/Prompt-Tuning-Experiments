import sys
sys.path.append("./../")
sys.path.append("./../trainer/")
from models.rnn_models import Seq2Seq
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
    "max_epochs": 1,
    "batch_size": 4,
    "seed": 7,
    "device": device
}

src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


model = Seq2Seq(
    enc_vocab_dim=config["src_vocab_size"],
    dec_vocab_dim=config["dst_vocab_size"],
    enc_hidden_dim=config["enc_hidden_dim"],
    dec_hidden_dim=config["dec_hidden_dim"],
    enc_n_layers=2,
    dec_n_layers=1,
    pad_idx=src_tokenizer.pad_token_id,
    start_idx=dst_tokenizer.sep_token_id,
    end_idx=dst_tokenizer.mask_token_id,
    teacher_forcing_ratio=0.5,
    device=device
)

trainer = Seq2SeqTrainer(model, src_tokenizer, dst_tokenizer, config)

trainer.train(lambda x: model.generate(x, max_len=200))
