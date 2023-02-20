import torch
from torch import nn
import torch.nn.functional as F
import random

'''
Implementation of the sequence2sequence model as descriped in the paper:
Neural Machine Translation by Jointly Learning to Align and Translate
(https://arxiv.org/abs/1409.0473)
'''

'''
Encoder of the sequence to sequence model implemented using a bidirectional GRU
'''
class Encoder(nn.Module):

    '''
    - input_dim: size of the vocabulary
    - hidden_dim: size of the embedding
    - n_layers: number of layers of the GRU
    '''
    def __init__(self, vocab_dim, enc_hidden_dim, dec_hidden_dim, n_layers) -> None:
        super(Encoder, self).__init__()
        self.input_dim = vocab_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.n_layers = n_layers
        self.embedder = nn.Embedding(vocab_dim, enc_hidden_dim)
        self.encoder = nn.GRU(enc_hidden_dim, enc_hidden_dim, n_layers, bidirectional=True)
        self.linear = nn.Linear(enc_hidden_dim*2, dec_hidden_dim)


    '''
    Forward pass of the encoder

    Input:
    - x: a tensor of size (length, batch_size) which contains the tokenized sentences

    Output:
    - out: the output of the GRU of size (length, batch_size, dec_hidden_dim)
    - hidden: the internal state of the last GRU layer of size (length, 2*enc_hidden_dim)
    '''
    def forward(self, x):
        
        '''
        Embedder receives as input a tensor (length, batch_size) and returns
        a tensor of size (length, batch_size, enc_hidden_dim)
        '''
        x = self.embedder(x)

        '''
        The GRU receives as input a tensor of size (length, batch_size, enc_hidden_dim)
        and returns two tensors:
        - The output for each input of size (length, batch_size, 2*enc_hidden_dim)
        - The hidden state of each layer of size (2*num_layers, batch_size, enc_hidden_dim)
        '''
        out, hidden = self.encoder(x)

        '''
        We concatenate the last hidden sates of the left-to-right and the right-to-left
        layers of the GRU to get a single hidden state tensor of size (length, 2*enc_hidden_dim)
        '''
        hidden_cat = torch.cat((hidden[-2, : ,:], hidden[-1, :, :]), dim=1)

        '''
        We use a linear layer with a tanh activation function to map the last hidden state of
        size (length, 2*enc_hidden_dim) to a tensor of size (length, dec_hidden_dim)
        '''
        hidden = torch.tanh(self.linear(hidden_cat))

        '''
        We return:
        - The output of the GRU of size (length, batch_size, hidden_dim)
        - The mapping of the internal hidden state of the GRU of size (length, dec_hidden_dim)
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
    Forward pass of the attention layer. The attention layers computes the energy scores e_{i,j}:
    e_{i,j} = V^T*tanh(W*[s_{i-1}|h_j])
    where s_{i-1} is the output of the encoder at time i-1 and h_j is the state of the encoder at time j
    Then it normalizes them them using the softmax to return the attention coefficients:
    alpha_{i,j} = exp(e_{i,j}) / sum_k(exp(e_{i,k}))

    Input:
    - dec_hidden: the hidden state of the decoder at time t-1
                  which is a tensor of size (batch_size, dec_hidden_dim) 
    - enc_output: the hidden states of the encoder at each time step
                  which is a tensor of size (length, batch_size, 2*enc_hidden_dim)

    Output:
    - attention: the attention scores of size (batch_size, length)
    '''
    def forward(self, dec_hidden, enc_output):

        # sequence length of the source
        src_len = enc_output.shape[0]

        '''
        dec_hidden state is repeated src_len times and it 
        becomes a tensor of shape (batch_size, length, dec_hidden_dim)
        '''
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)

        '''
        enc_ouput is reshaped from (length, batch_size, 2*enc_hidden_dim)
        to (batch_size, length, 2*enc_hidden_dim)
        '''
        enc_output = enc_output.permute(1, 0, 2)

        '''
        The output of the encoder and the repeated hidden state of the decoder
        to obtain a tensor of size (batch_size, length, 2*enc_hidden_dim + dec_hidden_dim)
        '''
        hidden_cat = torch.cat((dec_hidden, enc_output), dim = 2)
        
        '''
        First part of the energy computation: tanh(W*[s_{i-1}|h_j])
        which returns a tensor of size (batch_size, length, dec_hidden_dim) 
        '''
        energy = torch.tanh(self.score(hidden_cat))

        '''
        Final part of the energy computation: V^T * tanh(W*[s_{i-1}|h_j])
        It returns a tensor of size (batch_size, length, 1), therefore we use
        the squeeze to change te size to (batch_size, length)
        '''
        energy = self.v(energy).squeeze(2)

        '''
        We compute the alphas applying the softmax to the energy scores
        as a tensor of size (batch_size, length). Each alpha[i, :] is 
        a tensor of size (length) which are the attention coefficients
        for the i-th sentence of the batch with respect to the current
        state of the decoder
        '''
        alpha = F.softmax(energy, dim=1)

        '''
        We reshape alpha from (batch_size, length) to (batch_size, 1, length)
        '''
        alpha = alpha.unsqueeze(1)


        '''
        We compute the attentions scores which are a tensor of size
        (batch_size, 1, 2*enc_hidden_dim)
        '''
        attention = torch.bmm(alpha, enc_output)

        '''
        We reshape attention from (batch_size, 1, 2*enc_hidden_dim) to 
        (batch_size, 2*enc_hidden_dim)
        '''
        attention = attention.squeeze(1)


        return attention



'''
Decoder of the sequence to sequence model implemented using a GRU
'''
class Decoder(nn.Module):

    def __init__(self, vocab_dim, enc_hidden_dim, dec_hidden_dim, n_layers) -> None:
        super(Decoder, self).__init__()
        self.input_dim = vocab_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.n_layers = n_layers

        self.embedder = nn.Embedding(vocab_dim, dec_hidden_dim)
        self.decoder = nn.GRU(2*enc_hidden_dim  + dec_hidden_dim, dec_hidden_dim, n_layers, bidirectional=False)
        self.attention = AttentionLayer(enc_hidden_dim, dec_hidden_dim)
        self.fc_out = nn.Linear(2*(enc_hidden_dim  + dec_hidden_dim), vocab_dim)


    '''
    Forward pass of the decoder of the sequence to sequence model

    Input:
    - input: the ground-truth token that should be predicted by the decoder,
      which a tensor of shape (batch_size)

    - dec_hidden: the hidden state of the decoder at time t-1
      which is a tensor of size (batch_size, dec_hidden_dim) 

    - enc_output: the hidden states of the encoder at each time step
      which is a tensor of size (length, batch_size, 2*enc_hidden_dim)

    Output:
    - logits: the logits produced by the decoder which is a tensor of size (batch_size, voc_dim)
    - hidden: the next hidden state of the decoder of size (batch_size, dec_hidden_dim) 
    '''
    def forward(self, input, dec_hidden, enc_output):

        '''
        We reshape input from (batch_size) to (1, batch_size)
        '''

        '''
        We compute the word embedding of the input, which is a tensor of shape
        (batch_size, dec_hidden_dim)
        '''
        embedded = self.embedder(input)

        '''
        We compute the attention scores which are a tensor of shape (batch_size, 2*enc_hidden_dim)
        '''
        attention = self.attention(dec_hidden, enc_output)
        
        '''
        We concatenate the attention tensor with the  embedded tensor to get
        a tensor of size (batch_size, 2*enc_hidden_dim  + dec_hidden_dim)
        '''
        gru_input = torch.concat((embedded, attention), dim=1)

        '''
        We reshape gru_input from (batch_size, 2*enc_hidden_dim  + dec_hidden_dim)
        to (1, batch_size, 2*enc_hidden_dim  + dec_hidden_dim)
        '''
        gru_input = gru_input.unsqueeze(0)

        '''
        We reshape dec_hidden from (batch_size, dec_hidden_dim) to (1, batch_size, dec_hidden_dim) 
        '''
        dec_hidden = dec_hidden.unsqueeze(0)


        '''
        We pass to the GRU the tensor obtained concatenating the attention score and the
        encoder output, and the previous decoder hidden state, and we get to output tensors
        both of size (1, batch_size, dec_hidden_dim)
        '''
        output, hidden = self.decoder(gru_input, dec_hidden)


        '''
        We reshape output and hidden from (1, batch_size, dec_hidden_dim)
        to (batch_size, dec_hidden_dim)
        '''
        output = output.squeeze(0)
        hidden = hidden.squeeze(0)


        '''
        We concatenate the attention scores, the output of the GRU and the embedded of the
        input sequence to get a tensor of size (batch_size, 2*enc_hidden_dim  + 2*dec_hidden_dim)
        '''
        fc_input = torch.concat((output, attention, embedded), dim=1)

        '''
        We compute the logits which is a tensor of size (batch_size, voc_dim)
        '''
        logits = self.fc_out(fc_input)

        return logits, hidden



class Seq2Seq(nn.Module):


    def __init__(self, enc_vocab_dim, dec_vocab_dim, enc_hidden_dim, dec_hidden_dim, enc_n_layers, dec_n_layers, teacher_forcing_ratio, device) -> None:
        super(Seq2Seq, self).__init__()
        self.enc_vocab_dim = enc_vocab_dim
        self.dec_vocab_dim = dec_vocab_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_n_layers = enc_n_layers
        self.dec_n_layers = dec_n_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device

        self.encoder = Encoder(enc_vocab_dim, enc_hidden_dim, dec_hidden_dim, enc_n_layers)
        self.decoder = Decoder(dec_vocab_dim, enc_hidden_dim, dec_hidden_dim, dec_n_layers)


    '''
    Forward pass of the seq2seq model

    Input:
    - source: the input sentences, a tensor of size (src_len, batch_size)
    - target: the target sentences, a tensor of size (dst_len, batch_size)

    Output:
    - outputs: the logits for each token of each sentence,
               a tensor of size (dst_len, batch_size, dec_vocab_dim)
    '''
    def forward(self, source, target):
        
        # max sequence length of the target sentences
        target_len = target.shape[0]

        # batch size
        batch_size = target.shape[1]

        '''
        The source sentences are processed by the encoder that returns:
        - enc_output: a tensor of size (length, batch_size, 2*enc_hidden_dim)
        - hidden: a tensor of size (batch_size, dec_hidden_dim)
        '''
        enc_output, hidden = self.encoder(source)

        '''
        We prepare a tensor of size (dst_len, batch_size, dec_vocab_dim)
        that will store the output of the model
        '''
        outputs = torch.zeros(target_len, batch_size, self.dec_vocab_dim)

        '''
        We take the first token of each target sentence of the batch,
        a tensor of size (batch_size)
        '''
        target_token = target[0, :]

        # iterate over all the tokens of the target sentences
        for t in range(1, target_len):

            '''
            The decoder returns:
            - logits: the logits for the current target token, size (batch_size, dec_voc_dim)
            - hidden: the next hidden state of the decoder, size (batch_size, dec_hidden_dim) 
            '''
            logits, hidden = self.decoder(target_token, hidden, enc_output)

            outputs[t] = logits

            # decide whether to use teacher forcing or not
            teacher_force = random.uniform(0, 1) < self.teacher_forcing_ratio

            '''
            for each sentence we compute the most likely next token
            '''
            top1 = logits.argmax(1) 

            target_token = target[t, :] if teacher_force else top1

        return outputs