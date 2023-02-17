'''
test file to test how GRU layer works
'''


import torch
from torch import nn



batch_size = 64
seq_len = 20
in_dim = 10
hidden_dim = 5
n_layers = 2
# print(model)

# print("Model paramters:")

# # basically the input 

# for param in model.named_parameters():
#     print(param[0] + ":")
#     print(param[1].shape)
#     print("\n")


model = nn.GRU(in_dim, hidden_dim, n_layers, bidirectional=False)

x = torch.randn(batch_size, seq_len, in_dim)
hid = torch.randn(n_layers, seq_len, hidden_dim)
out, hidden = model(x, hid)

print("Output: ", out.shape)
print("Hidden: ", hidden.shape)
