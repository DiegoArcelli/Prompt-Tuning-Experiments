from transformers import T5Model, BartModel
import torch
from torch import nn
import torch.nn.functional as F
import random
from utils import get_model


'''
Class that implements a model for a neueral machine translation task
'''
class NMTModel(nn.Module):

    '''
    The constructor parameters are:
    - model: which is the encoder-decoder transformer to use for the NMT task
    - hidden_size: the dimension of the final output embedding produced by the docder of the model
    - voc_size: the vocabulary size of the target vocabulary
    '''
    def __init__(self, model, hidden_size, voc_size) -> None:
        super(NMTModel, self).__init__()
        self.model = model

        '''
        head is a linear layer which is used to map the final embedding of the decoder to a 
        a probability distribution over the vocabulary of the target language
        '''
        self.head = nn.Linear(hidden_size, voc_size, bias=False)

    
    '''
    Forward pass of the NMTModel

    Input:
    - inputs: which is the output produced by the tokenizer of the source language
    - targets (optional): which is the output produced by the tokenizer of the target language

    Output:
    - output: a dictionary that contains
        - last_hidden_state: the last hidden layer of the decoder of size (batch_size, dst_len, hidden_size) 
        - encoder_last_hidden_state: the last hidden layer of the decoder of size (batch_size, src_len, hidden_size) 
        - logits: the logits over the vocabulary of size (batch_size, dst_len, dst_voc_size) 
    '''
    def forward(self, inputs, targets=None):

        '''
        we extract from inputs the input ids of the sentences and the attention masks
        (in the masks 1 for all non padding tokens and 0 for padding tokens)
        '''
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        '''
        if targets is passed we extract from it the input ids and the attention masks of
        the target sentences. Otherwise we replicate those of inputs
        '''
        if targets is not None:
            target_input_ids = targets.input_ids
            target_attention_mask = targets.attention_mask
        else:
            target_input_ids = inputs.input_ids
            target_attention_mask = inputs.attention_mask

        '''
        the encoder-decoder model takes as input the input ids and the attention masks of
        the source sentences and of the target sentences
        '''
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_input_ids,
            decoder_attention_mask=target_attention_mask,
        )

        '''
        we get the last hidden state of the decoder which is a tensor
        of size (batch_size, dst_len, hidden_size)
        '''
        last_hidden_state = output.last_hidden_state

        '''
        we apply the model head to the last hidden state of the decoder to obtain
        the logits which is a tensor of size (batch_size, dst_len, dst_voc_size)
        '''
        output["logits"] = self.head(last_hidden_state)

        return output
    

'''
NMT model which uses T5 as encoder-decoder model
'''
class T5ForNMT(NMTModel):

    def __init__(self, hidden_size, voc_size) -> None:
        super(T5ForNMT, self).__init__(get_model("t5"), hidden_size, voc_size)
        

'''
NMT model which uses Bart as encoder-decoder model
'''
class BartForNMT(NMTModel):

    def __init__(self, hidden_size, voc_size) -> None:
        super(BartForNMT, self).__init__(get_model("bart"), hidden_size, voc_size)