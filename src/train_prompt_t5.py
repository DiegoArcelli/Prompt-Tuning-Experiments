import sys
sys.path.append("./../")
sys.path.append("./../trainer/")
from trainers.prompt_trainer import PromptTuningTrainer
from models.prompt_tuning_models import T5PromptTuning
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

model = T5PromptTuning.from_pretrained(
    "t5-small",
    encoder_soft_prompt_path = None,
    decoder_soft_prompt_path = None,
    encoder_n_tokens = 40,
    decoder_n_tokens = 40,
    encoder_hidden_dim=64,
    decoder_hidden_dim=64
)

trainer = PromptTuningTrainer(model, src_tokenizer, dst_tokenizer, config)

generate_fun = lambda x: model.generate(
    input_ids=x, 
    decoder_input_ids=torch.zeros([1,1]).long().to(config["device"]), 
    max_length=100,
    num_beams=5,
    early_stopping=True,
)

trainer.train(generate_fun)