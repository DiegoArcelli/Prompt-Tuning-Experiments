from transformers import T5Model, BartModel, T5ForConditionalGeneration
import torch
from torch import nn
import torch.nn.functional as F


'''
super class that defines the behavior of the T5 model with the soft-prompts
'''
class T5PromptTuningUtils:

    '''
    Wrapper of the from_pretrained class methods to include the loading of the soft-prompts
    '''
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        encoder_soft_prompt_path = None,
        decoder_soft_prompt_path = None,
        n_tokens = None,
        initialize_from_vocab = True,
        random_range = 0.5,
        device=None,
        **kwargs,
    ):
        # getting the T5 model
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freezing all the parameters of the pretrained T5 model
        for param in model.parameters():
            param.requires_grad = False

        '''
        load the encoder soft prompts if the path is provided otheriwise they
        are randomly initialized
        '''
        if encoder_soft_prompt_path is not None:
            model.set_encoder_soft_prompts(encoder_soft_prompt_path)
        else:
            model.initialize_encoder_soft_prompts(n_tokens, random_range)

        '''
        load the encoder soft prompts if the path is provided otheriwise they
        are randomly initialized
        '''
        if decoder_soft_prompt_path is not None:
            model.set_decoder_soft_prompts(decoder_soft_prompt_path)
        else:
            model.initialize_decoder_soft_prompts(n_tokens, random_range)

        return model
    

    def initialize_encoder_soft_prompts(self, n_tokens, random_range=0.5):
        self.n_tokens = n_tokens
        self.encoder_soft_prompt = nn.Embedding(n_tokens, self.config.d_model)
        init_prompt_value = torch.FloatTensor(2, 10).uniform_(-random_range, random_range)
        self.encoder_soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)


    def set_encoder_soft_prompts(self, soft_prompt_path):
        self.encoder_soft_prompt = torch.load(soft_prompt_path, map_location=torch.device("cpu"))
        self.n_tokens = self.encoder_soft_prompt.num_embeddings


    def initialize_decoder_soft_prompts(self, n_tokens, random_range=0.5):
        self.n_tokens = n_tokens
        self.decoder_soft_prompt = nn.Embedding(n_tokens, self.config.d_model)
        init_prompt_value = torch.FloatTensor(2, 10).uniform_(-random_range, random_range)
        self.decoder_soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)


    def set_decoder_soft_prompts(self, soft_prompt_path):
        self.decoder_soft_prompt = torch.load(soft_prompt_path, map_location=torch.device("cpu"))
        self.n_tokens = self.decoder_soft_prompt.num_embeddings


    def concatenate_encoder_soft_prompts(self, input_ids):
        embeddings = self.encoder.embed_tokens(input_ids.to(self.device))
        soft_prompts = self.encoder_soft_prompt.weight.repeat(embeddings.size(0), 1, 1)
        inputs_concat = torch.cat([soft_prompts, embeddings], dim=1)
        return inputs_concat
    

    def concatenate_decoder_soft_prompts(self, input_ids):
        embeddings = self.decoder.embed_tokens(input_ids.to(self.device))
        soft_prompts = self.decoder_soft_prompt.weight.repeat(embeddings.size(0), 1, 1)
        inputs_concat = torch.cat([soft_prompts, embeddings], dim=1)
        return inputs_concat


    def extend_attention_mask(self, attention_mask):
        batch_size = attention_mask.shape[0]
        soft_prompts_mask = torch.full((batch_size, self.n_tokens), 1, dtype=torch.long).to(self.device)
        extended_mask = torch.concat([soft_prompts_mask, attention_mask], dim=1)
        return attention_mask.to(self.device)



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
        labels=None,
        return_dict=None,
        *args,
        **kwargs
    ):
        
        if input_ids is not None:
            inputs_embeds = self.concatenate_encoder_soft_prompts(input_ids)
            input_ids = None

        if decoder_input_ids is not None:
            decoder_input_ids = self.extend_attention_mask(decoder_input_ids)

        if attention_mask is not None:
            attention_mask = self.extend_attention_mask(attention_mask)

        if decoder_attention_mask is not None:
            decoder_attention_mask = self.extend_attention_mask(attention_mask)


        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            *args,
            **kwargs
        )


'''
Defining the T5 model with prompt tuning superclassing T5PromptTuningUtils and 
T5ForConditionalGeneration (which adds the head for producing the logits)
'''
class T5PromptTuning(T5PromptTuningUtils, T5ForConditionalGeneration):

    def __init__(self, config) -> None:
        super(T5PromptTuning, self).__init__(config)