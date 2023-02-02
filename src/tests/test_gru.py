'''
test file to test how GRU layer works
'''


import torch
from torch import nn


model = nn.GRU(10, 5, 3, bidirectional=False)
print(model)

print("Model paramters:")

# basically the input 

for param in model.named_parameters():
    print(param[0] + ":")
    print(param[1].shape)
    print("\n")


x = torch.randn(20, 10)
y = model(x)

# 
print(y[0].shape)
print(y[1].shape)
