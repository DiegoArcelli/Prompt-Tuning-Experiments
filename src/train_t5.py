import sys
sys.path.append("./../")
sys.path.append("./../trainer/")
from models.nmt_models import T5ForNMT
from trainers.seq2seq_trainer import Trainer
from models.prompt_tuning_models import T5PromptTuning
from transformers import AutoTokenizer
import torch

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


src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# model = T5ForNMT.from_pretrained("t5-small", hidden_size=512, voc_size=28996)
model = T5PromptTuning.from_pretrained(
    "t5-small",
    encoder_soft_prompt_path = None,
    decoder_soft_prompt_path = None,
    encoder_n_tokens = 20,
    decoder_n_tokens = 20,
    encoder_hidden_dim=64,
    decoder_hidden_dim=64
)

trainer = Trainer(model, src_tokenizer, dst_tokenizer, config)

generate_fun = lambda x: model.generate(
    input_ids=x, 
    decoder_input_ids=torch.zeros([1,1]).long(), 
    max_length=200,
    num_beams=5,
    early_stopping=True,
)

trainer.train(generate_fun)