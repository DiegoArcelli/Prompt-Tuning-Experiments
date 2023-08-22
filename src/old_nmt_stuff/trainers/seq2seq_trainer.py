from sys import path
path.append("./trainers")
from trainer_constants import *
import torch
from torch import nn
from torch import optim
from trainer import Trainer
from tqdm import tqdm


class Seq2SeqTrainer(Trainer):

    def __init__(self, model, src_tokenizer, dst_tokenizer, config) -> None:
        super(Seq2SeqTrainer, self).__init__(model, src_tokenizer, dst_tokenizer, config)

        pad_token = dst_tokenizer.pad_token
        pad_token_idx = dst_tokenizer.convert_tokens_to_ids([pad_token])[0]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)



    def train_step(self, train_loader, epoch):

        total_loss = 0
        n = len(train_loader)

        with tqdm(total=n) as pbar:
            for step, batch in enumerate(train_loader):

                self.optimizer.zero_grad()

                inputs, targets = batch
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

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

                if (step+1) % 10 == 0:
                    print(f"\nEpoch {epoch}, samples {step+1}/{n} train loss: {total_loss/(step+1)}")

                pbar.update(1)

        avg_loss = total_loss / n
        

        return avg_loss
    


    def val_step(self, val_loader, epoch):
        
        total_loss = 0
        n = len(val_loader)

        with tqdm(total=n) as pbar:
            for step, batch in enumerate(val_loader):

                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

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

                total_loss += loss.item()

                if (step+1) % 10 == 0:
                    print(f"\nEpoch {epoch}, samples {step+1}/{n} validation loss: {total_loss/(step+1)}")

                pbar.update(1)

        avg_loss = total_loss / n

        return avg_loss
    


    def test_step(self, test_loader):

        self.model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/model_{self.model_name}_{self.best_epoch}_checkpoint.pt"))
        
        total_loss = 0

        n = len(test_loader)

        with tqdm(total=n) as pbar:

            for step, batch in enumerate(test_loader):

                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

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

                total_loss += loss.item()

                pbar.update(1)

        avg_loss = total_loss / len(test_loader)

        return avg_loss