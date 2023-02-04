from models import BartForNMT
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

sentence = "I love my cute dog"

model = BartForNMT(768, 32768)

inputs = tokenizer(sentence, return_tensors="pt")

output = model(inputs.input_ids)
print(output.keys())