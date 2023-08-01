from models.prompt_tuning_models import T5PromptTuningSimple
from utils import count_parameters
from transformers import AutoTokenizer
import torch

model = T5PromptTuningSimple.from_pretrained("t5-small", None, None, 40, 40)
print(model)

tokenizer = AutoTokenizer.from_pretrained("t5-small")

inputs = tokenizer(["Hello, my dog is cute", "I hate black cats"], padding=True, truncation=True, return_tensors="pt")
targets = tokenizer(["Ciao, il mio cane è carino", "Odio i gatti neri"], padding=True, truncation=True, return_tensors="pt")

print(inputs.input_ids.shape, targets.input_ids.shape)

out = model(input_ids=inputs.input_ids, decoder_input_ids = targets.input_ids)
print(out.keys())
logits = out.logits
print(logits.shape)

for n, p in model.named_parameters():
    if p.requires_grad:
         print(n, p.shape)