from transformers import BartModel
from torch import nn


class BartForNMT(nn.Module):

    def __init__(self, hidden_size, voc_size) -> None:
        super(BartForNMT, self).__init__()
        self.bart = BartModel.from_pretrained("facebook/bart-base")
        self.head = nn.Linear(hidden_size, voc_size, bias=False)

    def forward(self, inputs):
        output = self.bart(inputs)
        last_hidden_state = output.last_hidden_state
        output["logits"] = self.head(last_hidden_state)
        return output
        


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers) -> None:
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.embedder = nn.Embedding(input_dim, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, output_dim, n_layers, bidirectional=True)


    def forward(self, x):
        x = self.embedder(x)
        out, hidden = self.encoder(x)
        return out, hidden
        