import torch
from torch import nn
import torch.functional as F
from models import Decoder, Encoder, Seq2Seq, AttentionLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 5000
seq_len = 20
batch_size = 64

# x = torch.randint(0, 5000, (20, 64))
# y = torch.randint(0, 5000, (batch_size, 20))
# ctx = torch.randn(64, 20, 256)

# enc = Encoder(vocab_size, 256, 4)
# out, hidden = enc(x)
# print(out.shape)
# print(hidden.shape)

# inp = torch.randint(0, vocab_size, (batch_size,))
# x = torch.randn(batch_size, 128)
# y = torch.randn(seq_len, batch_size, 256)

# dec = Decoder(vocab_size, 128, 128, 1)

# logits, hidden = dec(inp, x, y)
# print(logits.shape)
# print(hidden.shape)

model = Seq2Seq(5000, 6000, 256, 256, 4, 1, 0.5, device)



x = torch.randint(0, 5000, (seq_len, batch_size))
y = torch.randint(0, 6000, (seq_len-3, batch_size))


print(x.shape)
print(y.shape)

out = model(x, y)
print(out.shape)