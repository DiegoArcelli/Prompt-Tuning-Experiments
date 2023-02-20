import sys
sys.path.append("./../")
from trainer import Trainer
from models.rnn_models import Seq2Seq
from models.nmt_models import BartForNMT, T5ForNMT
from transformers import AutoTokenizer

model = T5ForNMT(512, 32100)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# model = BartForNMT(768, 32100)
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

prompt = ["i love my dog"]
input_ids = tokenizer(prompt, return_tensors='pt')


out = model(input_ids)
print(out.keys())
print(out["logits"].shape)