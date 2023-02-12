import torch
from torch import nn
import torch.functional as F
from models import Decoder, Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 5000
batch_size = 100

x = torch.randint(0, 5000, (batch_size, 20))
y = torch.randint(0, 5000, (batch_size, 20))

enc = Encoder(vocab_size, 256, 1)
dec = Decoder(vocab_size, 256, 1, 1)

c, _ = enc(x)

print("Encoder output", c.shape)

logits, state = dec(y, c)
print("Decoder state: ", state.shape)
print("Decoder output: ", logits.shape)