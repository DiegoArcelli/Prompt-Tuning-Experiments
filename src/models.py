from transformers import BartModel
import torch
from torch import nn
import torch.nn.functional as F

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
    - n_layers: number of layers of the GRU
    '''
    def __init__(self, vocab_dim, hidden_dim, n_layers) -> None:
        super(Encoder, self).__init__()
        self.input_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedder = nn.Embedding(vocab_dim, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, n_layers, bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, hidden_dim)


    '''
    Forward pass of the encoder

    Input:
    - x: a tensor of size (batch_size, length) which contains the tokenized sentences

    Output:
    - out: the output of the GRU of size (batch_size, length, hidden_dim)
    - hidden: the internal state of the last GRU layer of size (length, hidden_dim)
    '''
    def forward(self, x):

        '''
        Embedder receives as input a tensor (batch_size, length) and returns
        a tensor of size (batch_size, length, hidden_dim)
        '''
        x = self.embedder(x)

        '''
        The encoder receives as input a tensor of size (batch_size, length, hidden_dim)
        and returns two tensors:
        - The output for each input of size (batch_size, length, 2*hidden_dim)
        - The hidden state of each layer of size (2*num_layers, length, hidden_dim)
        '''
        out, hidden = self.encoder(x)

        '''
        We concatenate the last hidden sates of the left-to-right and the right-to-left
        layers of the GRU to get a single hidden state tensor of size (length, 2*hidden_dim)
        '''
        hidden_cat = torch.cat((hidden[-2, : ,:], hidden[-1, :, :]), dim=1)

        '''
        We use a linear layer with a tanh activation function to map the last hidden state of
        size (length, 2*hidden_dim) to a tensor of size (length, hidden_dim)
        '''
        hidden = torch.tanh(self.linear(hidden_cat))

        '''
        We return:
        - The output of the GRU of size (batch_size, length, hidden_dim)
        - The mapping of the internal hidden state of the GRU of size (length, hidden_dim)
        '''
        return out, hidden


class AttentionLayer(nn.Module):
    
    def __init__(self, enc_hidden_dim, dec_hidden_dim) -> None:
        super(AttentionLayer, self).__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        
        self.score = nn.Linear(2*enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias = False)


    '''
    Forward pass of the attention layer

    Input:
    - dec_hidden: the hidden state of the decoder at time t-1
                  which is a tensor of size (batch_size, dec_hidden_dim) 
    - enc_output: the hidden states of the encoder at each time step
                  which is a tensor of size (batch_size, length, 2*enc_hidden_dim)


    Output:

    '''
    def forward(self, dec_hidden, enc_output):

        # sequence length of the source
        src_len = enc_output.shape[1]

        '''
        dec_hidden state is repreated src_len times and it 
        becomes a tensor of shape (batch_size, length, length, dec_hidden_dim)
        '''
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)


        '''
        we compute the scores of each encoder hidden state with the corresponding 
        '''
        hidden_cat = torch.tanh( torch.cat((dec_hidden, enc_output), dim = 2) )
        energy = self.score(hidden_cat)
        attention = self.v(energy).squeeze(2)
        norm_attention =  F.softmax(attention, dim=1)

        return norm_attention



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
        print(out.shape, weights.shape)
        logits = self.output(out)

        return logits, state



class Seq2Seq(nn.Module):


    def __init__(self, enc_vocab_dim, dec_vocab_dim, enc_hidden_dim, dec_hidden_dim, enc_n_layers, dec_n_layers, n_heads) -> None:
        super(Seq2Seq, self).__init__()
        self.enc_vocab_dim = enc_vocab_dim
        self.dec_vocab_dim = dec_vocab_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_n_layers = enc_n_layers
        self.dec_n_layers = dec_n_layers
        self.n_heads =  n_heads

        self.encoder = Encoder(enc_vocab_dim, enc_hidden_dim, enc_n_layers)
        self.decoder = Decoder(dec_vocab_dim, dec_hidden_dim, dec_n_layers, n_heads)

    def forward(self, x, y):
        context, _ = self.encoder(x)
        logits, state = self.decoder(y, context)
        return logits, state