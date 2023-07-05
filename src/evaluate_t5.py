import sys
sys.path.append("./../")
sys.path.append("./../trainer/")
from trainers.evaluator import Evaluator
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

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
    "device": device,
    "lang": "deu",
    "prefix": True,
}

generate_fun = lambda x: model.generate(
    **x, 
)

eval = Evaluator(model, tokenizer, tokenizer, config)
eval.train(generate_fun)