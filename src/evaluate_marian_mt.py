import sys
sys.path.append("./../")
sys.path.append("./../trainer/")
from trainers.evaluator import Evaluator
from transformers import MarianTokenizer, MarianMTModel
import torch

model_name = "Helsinki-NLP/opus-mt-tc-big-en-it"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

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
    "device": device
}

generate_fun = lambda x: model.generate(
    **x, 
)

eval = Evaluator(model, tokenizer, tokenizer, config)
eval.train(generate_fun)