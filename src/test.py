from models import BartForNMT, Encoder
from transformers import AutoTokenizer
import torch

# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# sentence = ["I love my cute dog",
#             "I hate black cats",
#             "I eat candies all day",
#             "I don't want to set the world on fire"
#             ]

# # model = BartForNMT(768, 32768)

# inputs = tokenizer(sentence, return_tensors="pt", max_length=20, padding='max_length').input_ids
# print(inputs.shape)
# output = model(inputs.input_ids)
# print(output.keys())
x = torch.randint(0, 5000, (100, 20))
enc = Encoder(5000, 256, 128, 5)
y, h = enc(x)

print("Input: ", x.shape)
print("Output: ", y.shape)
print("Hidden: ", h.shape)
