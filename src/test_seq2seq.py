import torch
from torch import nn
import torch.functional as F
from models import Decoder, Encoder, Seq2Seq, AttentionLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 5000
batch_size = 100

# x = torch.randint(0, 5000, (batch_size, 20))
# y = torch.randint(0, 5000, (batch_size, 20))
# ctx = torch.randn(64, 20, 256)


# enc = Encoder(vocab_size, 256, 4)
# dec = Decoder(vocab_size, 256, 1, 1)
# attn = AttentionLayer(256, 1)

# c, h = enc(x)
# print(c.shape)
# print(h.shape)

# print("Encoder output", c.shape)

# logits, state = dec(y, c)
# print("Decoder state: ", state.shape)
# print("Decoder output: ", logits.shape)


# model = Seq2Seq(vocab_size, vocab_size, 256, 512, 1, 1, 1)
# logits, state = model(x, y)
# print("Logits: ", logits.shape)
# print("State: ", state.shape)

x = torch.randn(batch_size, 128)
y = torch.randn(batch_size, 20, 256)

attn = AttentionLayer(128, 128)

attn(x, y)
