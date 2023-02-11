from transformers import BartModel, BartForConditionalGeneration
from transformers import BartTokenizer
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


model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
# model = BartForNMT(768, 50265)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

sentences = ("I love my cute dog",
            "I hate black cats",
            "I was walking down the street",
            "I hate the knee grow",
            "Please don't say this")

inputs = tokenizer(sentences, return_tensors="pt", padding=True)

output = model(inputs.input_ids)
# print(output.keys())
print(output.logits.shape) # size n_batch x n_tokens x logits