import sys
sys.path.append("./../")
sys.path.append("./../trainer/")
from nmt_datasets import AnkiDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from models.nmt_models import T5ForNMT
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.prompt_tuning_models import T5PromptTuning
import time
from tqdm import tqdm
from utils import count_parameters
from transformers import MT5ForConditionalGeneration, AutoTokenizer, T5ForConditionalGeneration
import evaluate
from models.nmt_models import MT5ForNMT

# tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
# model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
# model = MT5ForNMT.from_pretrained("google/mt5-small")

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

source_sentences = ["translate English to German: I love you", "Mi chiamo diego", "Odio i cani"]
target_sentences = ["Hello how are you?", "My name is diego", "I hate dogs"]
# test_sentences = [f"{prefix}: {sentence}" for sentence in test_sentences]
inputs_src = tokenizer(source_sentences, padding=True, truncation=True, return_tensors="pt")
inputs_tar = tokenizer(target_sentences, padding=True, truncation=True, return_tensors="pt")
print(inputs_src.keys())
# outputs = model(input_ids=inputs_src.input_ids, labels=inputs_tar.input_ids)
outputs = model(input_ids=inputs_src.input_ids, decoder_input_ids=torch.zeros([3,1]).long())

print(outputs.keys())
print()
print(inputs_src.input_ids.shape)
print(inputs_tar.input_ids.shape)
print(outputs.logits.shape)
# print(outputs.loss)

gen = model.generate(
    input_ids=inputs_src.input_ids[:1, :], 
    decoder_input_ids=torch.zeros([1, 1]).long(),
    early_stopping=True,
)
print(gen)

print(tokenizer.batch_decode(gen, skip_special_tokens=True))