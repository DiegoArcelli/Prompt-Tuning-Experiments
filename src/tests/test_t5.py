from transformers import BartModel, T5Model, T5ForConditionalGeneration
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

prompt = "summarize: i love my dog really much and I will keep it for the rest of my life"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

out = model.generate(input_ids, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
print(tokenizer.convert_ids_to_tokens(out[0]))