from torch.utils.data import Dataset, RandomSampler
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import BartModel
import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split
import random
import numpy as np
import time
from trainer import Trainer


class Seq2SeqTrainer(Trainer):

    def __init__(self, model, src_tokenizer, dst_tokenizer, config) -> None:
        super(Seq2SeqTrainer, self).__init__(model, src_tokenizer, dst_tokenizer, config)

        pad_token = dst_tokenizer.pad_token
        pad_token_idx = dst_tokenizer.convert_tokens_to_ids([pad_token])[0]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)



    def train_step(self, train_loader):

        total_loss = 0

        for step, batch in enumerate(train_loader):

            self.optimizer.zero_grad()

            inputs, targets = batch

            '''
            reshape input tensors from (batch_size, length) to (length, batch_size)
            '''
            input_ids = inputs.input_ids.permute(1, 0)
            target_ids = targets.input_ids.permute(1, 0)

            output = self.model(input_ids, target_ids)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            target_ids = target_ids[1:].reshape(-1)

            loss = self.criterion(output, target_ids)
            
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(train_loader)

        return avg_loss
    

    def val_step(self, val_loader):
        
        total_loss = 0

        for step, batch in enumerate(val_loader):

            self.optimizer.zero_grad()

            inputs, targets = batch

            '''
            reshape input tensors from (batch_size, length) to (length, batch_size)
            '''
            input_ids = inputs.input_ids.permute(1, 0)
            target_ids = targets.input_ids.permute(1, 0)

            output = self.model(input_ids, target_ids)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            target_ids = target_ids[1:].reshape(-1)

            loss = self.criterion(output, target_ids)
            
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(val_loader)

        return avg_loss