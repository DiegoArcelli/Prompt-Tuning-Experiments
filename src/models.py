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
        

'''
Encoder of the seq2seq model
It uses and Embedding layer to map the words of each input sentence to vector
and then the embedding are passed to a Bidirectional GRU to produce 
the context vector for the decoder
'''
class Encoder(nn.Module):

    '''
    - input_dim: size of the vocabulary
    - hidden_dim: size of the embedding
    '''
    def __init__(self, vocab_dim, hidden_dim, n_layers) -> None:
        super(Encoder, self).__init__()
        self.input_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedder = nn.Embedding(vocab_dim, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, n_layers, bidirectional=False)


    def forward(self, x):
        x = self.embedder(x)
        out, hidden = self.encoder(x)
        return out, hidden
    


class AttentionLayer(nn.Module):

    
    def __init__(self, embedded_dim, n_heads) -> None:
        super(AttentionLayer, self).__init__()
        self.mha = nn.MultiheadAttention(embedded_dim, n_heads)

    def forward(self, input, context):
        out, weights = self.mha(context, input, input)
        return out, weights



class Decoder(nn.Module):

    def __init__(self, vocab_dim, hidden_dim, n_layers, n_heads) -> None:
        super(Decoder, self).__init__()
        self.input_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.embedder = nn.Embedding(vocab_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, n_layers, bidirectional=False)
        self.attention = AttentionLayer(hidden_dim, n_heads)
        self.output = nn.Linear(hidden_dim, vocab_dim)

    def forward(self, x, context):
        x = self.embedder(x)
        x, state = self.decoder(x)
        out, weights = self.attention(x, context)
        logits = self.output(out)

        return logits, state



class Seq2Seq(nn.Module):


    def __init__(self, enc_vocab_dim, dec_vocab_dim, hidden_dim, enc_n_layers, dec_n_layers, n_heads) -> None:
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(enc_vocab_dim, hidden_dim, enc_n_layers)
        self.decoder = Decoder(dec_vocab_dim, hidden_dim, dec_n_layers, n_heads)

    def forward(self, x, y):
        context, _ = self.encoder(x)
        logits, state = self.decoder(y, context)
        return logits, state