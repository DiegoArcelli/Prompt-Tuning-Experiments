from transformers import BartModel, T5Model, T5ForConditionalGeneration
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('t5-small')
# model = T5ForConditionalGeneration.from_pretrained('t5-small')

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

source = "I love my dog"
inputs = tokenizer(source, return_tensors='pt')

target = "Io amo il mio cane"
targets = tokenizer(target, return_tensors='pt')

input_ids = inputs.input_ids
target_ids = targets.input_ids

print(len(input_ids[0]), len(target_ids[0]))

out = model(inputs.input_ids, labels=targets.input_ids)
print(out.keys())
logits = out.logits
print(logits.shape)


# out = model.generate(input_ids, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
# print(tokenizer.convert_ids_to_tokens(out[0]))

# out = model(inputs.input_ids)
print(model.encoder.get_input_embeddings().weight.shape[1])
print(model.decoder.get_input_embeddings().weight.shape[1])