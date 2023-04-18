from sys import path
path.append("./trainers")
from trainer_constants import *
import torch
from torch import nn
from torch import optim
from trainer import Trainer
from tqdm import tqdm


class PromptTuningTrainer(Trainer):


    def __init__(self, model, src_tokenizer, dst_tokenizer, config) -> None:
        super(PromptTuningTrainer, self).__init__(model, src_tokenizer, dst_tokenizer, config)

        prefixes = [
            "encoder_soft_prompt",
            "decoder_soft_prompt",
            "encoder_emb_generator",
            "decoder_emb_generator"
            ]
        
        prompt_parameters = [p for n, p in model.named_parameters() if any([n.startswith(prefix) for prefix in prefixes])]
        self.optimizer = optim.Adam(prompt_parameters, lr=0.1)
    

    
    def train_step(self, train_loader, epoch):

        total_loss = 0
        n = len(train_loader)

        with tqdm(total=n) as pbar:
            for step, batch in enumerate(train_loader):

                self.optimizer.zero_grad()
                inputs, targets = batch

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                input_ids = inputs.input_ids
                target_ids = targets.input_ids

                output = self.model(input_ids=input_ids, decoder_input_ids=target_ids)

                target_ids = self.model.extend_labels(target_ids, self.pad_token_idx)

                logits = output.logits

                logits_dim = logits.shape[-1]

                logits = logits[1:].view(-1, logits_dim)
                target_ids = target_ids[1:].reshape(-1)

                loss = self.criterion(logits, target_ids)
                
                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

                if (step+1) % 100 == 0:
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

                input_ids = inputs.input_ids
                target_ids = targets.input_ids

                output = self.model(input_ids=input_ids, decoder_input_ids=target_ids)

                target_ids = self.model.extend_labels(target_ids, self.pad_token_idx)

                logits = output.logits

                logits_dim = logits.shape[-1]

                logits = logits[1:].view(-1, logits_dim)
                target_ids = target_ids[1:].reshape(-1)

                loss = self.criterion(logits, target_ids)

                total_loss += loss.item()

                if (step+1) % 100 == 0:
                    print(f"\nEpoch {epoch}, samples {step+1}/{n} train loss: {total_loss/(step+1)}")

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

                input_ids = inputs.input_ids
                target_ids = targets.input_ids

                output = self.model(input_ids=input_ids, decoder_input_ids=target_ids)

                target_ids = self.model.extend_labels(target_ids, self.pad_token_idx)

                logits = output.logits

                logits_dim = logits.shape[-1]

                logits = logits[1:].view(-1, logits_dim)
                target_ids = target_ids[1:].reshape(-1)

                loss = self.criterion(logits, target_ids)

                total_loss += loss.item()

                pbar.update(1)


        avg_loss = total_loss / n

        return avg_loss