from transformers import T5Model, BartModel
import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Model
from transformers.utils import logging


logger = logging.get_logger(__name__)

'''
Superclass which is used to add a linear head in order to produce the probability
distribution over the words of the vocabulary, in an auto-regressive way
'''
class NMTModelMixin:

    '''
    class method to load a pre-trained language model and add a classification head on top of it

    Inputs:
    - cls: the class reference
    - pretrained_model_name_or_path: name of the pre-trained model in the hugging face repository
    - hidden_size: dimension of the hidden representation used by the pre-trained model
    - voc_size: vocabulary size of the model

    Output:
    - model: the pre-trained language model with the linear head added on top of it
    '''
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        hidden_size,
        voc_size,
        **kwargs,
    ):
        
        '''
        loading the pre-trained model identified by pretrained_model_name_or_path
        '''
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        '''
        head is a linear layer which is used to map the final embedding of the decoder to a 
        a probability distribution over the vocabulary of the target language
        '''
        model.head = nn.Linear(hidden_size, voc_size, bias=False)

        return model
    

    '''
    definition of the forward layer of the model (the parameters are the
    same of a huggingface sequence-to-sequence model)
    '''
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_inputs_embeds=None,
        encoder_outputs=None,
        use_cache=None,
        return_dict=None,
        *args,
        **kwargs
    ):
        
        '''
        we call the forward layer of the super class (so of the )
        '''
        output =  super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            *args,
            **kwargs
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
    method needed for text generation
    '''
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    

    '''
    method needed for text generation
    '''
    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


'''
NMT model which uses T5 as encoder-decoder model
'''
class T5ForNMT(NMTModelMixin, T5Model):

    def __init__(self, config) -> None:
        super(T5ForNMT, self).__init__(config)