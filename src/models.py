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

    def __init__(self) -> None:
        super().__init__()