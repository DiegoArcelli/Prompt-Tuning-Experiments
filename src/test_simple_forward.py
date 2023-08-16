from models.prompt_tuning_models import MT5PromptTuningSimple
from utils import count_parameters
from transformers import AutoTokenizer, MT5ForConditionalGeneration
import torch

model = MT5PromptTuningSimple.from_pretrained("google/mt5-small", None, None, 10, 10)

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

inputs = tokenizer(["Hello, my dog is cute", "I hate black cats"], padding=True, truncation=True, return_tensors="pt")
targets = tokenizer(["Ciao, il mio cane è carino", "Odio i gatti neri"], padding=True, truncation=True, return_tensors="pt")

out = model(input_ids=inputs.input_ids, decoder_input_ids = targets.input_ids, attention_mask=inputs.attention_mask, decoder_attention_mask=targets.attention_mask)
print(out.keys())
logits = out.logits
print(logits.shape)