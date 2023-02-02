'''
test file to test how Embedding layer works
'''

from torch import nn
import torch

'''
The embedding layer maps an intenger number in the range [0, VOC_SIZE-1] to 
a vector of size 
'''

VOC_SIZE = 50 # the size of the vocabulary
EMB_SIZE = 5 # the size of the embeded words

model = nn.Embedding(VOC_SIZE, EMB_SIZE)
params = list(model.named_parameters())

print("Parameters of the model:")
W = params[0][1] # VOC_SIZE X EMB_SIZE matrix
print(W.shape)

print("\n")

x = torch.LongTensor([1, 5, 35, 49])
y = model(x)

for i in range(len(x)):
    print(f"Embedding for word of the vocabulary {x[i]}:\n{y[i]}\n")
    print(f"Correspoding matrix row:\n{y[i]}\n\n")
