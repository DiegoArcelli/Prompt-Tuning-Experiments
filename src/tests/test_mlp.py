import sys
sys.path.append("./../")
import torch
from torch import nn

emb = nn.Embedding(10, 20)
prompts = nn.Embedding(20, 20)

mlp = nn.Sequential(
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 40),
    nn.ReLU()
)

y = mlp(emb.weight)

print(emb.weight.shape)
print(y.shape)
