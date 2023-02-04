from transformers import BartModel, T5Model, T5ForConditionalGeneration
from transformers import AutoTokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

sentence = "I love my cute dog"

inputs = tokenizer(sentence, return_tensors="pt")

output = model(inputs.input_ids)
print(output.keys())
