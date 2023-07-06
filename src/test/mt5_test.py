import sys
sys.path.append("./../")
sys.path.append("./../trainer/")
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5Tokenizer, T5ForConditionalGeneration
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
from transformers import MT5ForConditionalGeneration, AutoTokenizer
import evaluate

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

# training
# input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
# labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
# outputs = model(input_ids=input_ids, labels=labels)
# loss = outputs.loss
# logits = outputs.logits

# inference
input_ids = tokenizer(
    "translate English to Italian: I like pizza", return_tensors="pt"
).input_ids  # Batch size 1
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# studies have shown that owning a dog is good for you.