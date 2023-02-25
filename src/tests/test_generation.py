from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

sentence = "I want to break free"

print(sentence)

inputs = tokenizer(sentence, return_tensors="pt")

# output = model(inputs.input_ids)
# logits = output["logits"]
# tokens = torch.argmax(logits, dim=2)
# print(tokenizer.decode(tokens[0], skip_special_tokens=True))

output = model.greedy_search(inputs.input_ids, max_length=100)
print(output)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# output = model.generate(
#     inputs.input_ids,
#     max_length=50, 
#     num_beams=5, 
#     early_stopping=True
# )

# print(tokenizer.decode(output[0], skip_special_tokens=True))
