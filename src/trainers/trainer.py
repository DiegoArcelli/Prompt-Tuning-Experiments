from sys import path
path.append("./../")
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import BartModel
import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split
import random
import numpy as np
from nmt_datasets import AnkiDataset
from utils import plot_curves
import os
from trainer_constants import *
import evaluate
from tqdm import tqdm


class Trainer:

    def __init__(self, model, src_tokenizer, dst_tokenizer, config) -> None:

        self.device = config["device"]
        self.model = model.to(self.device)
        self.src_tokenizer = src_tokenizer
        self.dst_tokenizer = dst_tokenizer
        self.config = config

        pad_token = dst_tokenizer.pad_token
        pad_token_idx = dst_tokenizer.convert_tokens_to_ids([pad_token])[0]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        self.pad_token_idx = pad_token_idx

        self.metric = evaluate.load("bleu")

        if "model_name" in config:
            self.model_name = config["model_name"]
        else:
            self.model_name = self.model.__class__.__name__.lower()


    
    def set_seeds(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


    def get_data_loader(self, batch_size, val_split=0.2, test_split=0.1):
        
        data_set = AnkiDataset(
            f"{DATASET_PATH}/ita.txt",
            self.src_tokenizer,
            self.dst_tokenizer,
            self.config["src_max_length"],
            self.config["dst_max_length"]
        )


        n = len(data_set)

        val_size = int(n*val_split)
        test_size = int(n*test_split)
        train_size = n - val_size - test_size


        train_set, val_set, test_set = random_split(data_set, [train_size, val_size, test_size])

        train_loader = DataLoader(
                    train_set,
                    batch_size = batch_size
                )
        
        val_loader = DataLoader(
                    val_set,
                    batch_size=batch_size
                )
        
        test_loader = DataLoader(
                    test_set,
                    batch_size = batch_size
                )
        
        return train_loader, val_loader, test_loader


    def generate_learning_curvers(self, train_losses, val_losses):

        plot_curves(
            curve_1=train_losses,
            curve_2=val_losses,
            label_1="Train loss",
            label_2="Validation loss",
            fig_name=f"{IMAGE_PATH}/loss_model_{self.model_name}"
        )

        plot_curves(
            curve_1=train_losses[:self.best_epoch],
            curve_2=val_losses[:self.best_epoch],
            label_1="Train loss",
            label_2="Validation loss",
            fig_name=f"{IMAGE_PATH}/best_loss_model_{self.model_name}"
        )

        plot_curves(
            curve_1=train_losses,
            label_1="Train loss",
            fig_name=f"{IMAGE_PATH}/train_loss_model_{self.model_name}"
        )

        plot_curves(
            curve_1=train_losses[:self.best_epoch],
            label_1="Train loss",
            fig_name=f"{IMAGE_PATH}/best_train_loss_model_{self.model_name}"
        )

        plot_curves(
            curve_1=val_losses,
            label_1="Val loss",
            fig_name=f"{IMAGE_PATH}/val_loss_model_{self.model_name}"
        )

        plot_curves(
            curve_1=val_losses[:self.best_epoch],
            label_1="Val loss",
            fig_name=f"{IMAGE_PATH}/best_val_loss_model_{self.model_name}"
        )


    def train(self, generate_fun):
        
        seed = self.config["seed"]
        self.set_seeds(seed)

        batch_size = self.config["batch_size"]
        # self.model.to(self.device)

        train_loader, val_loader, test_loader = self.get_data_loader(batch_size, 0.2, 0.1)

        self.train_loop(train_loader, val_loader)
        self.model.eval()
        test_loss = self.test_step(test_loader)
        print("Evaluating model on the test set")
        print(f"Test loss: {test_loss}")

        # evaluate bleu score
        train_score = self.metric_evaluation(train_loader, generate_fun)
        val_score = self.metric_evaluation(val_loader, generate_fun)
        test_score = self.metric_evaluation(test_loader, generate_fun)

        print(f"Average train set BLEU score: {train_score}")
        print(f"Average validation set BLEU score: {val_score}")
        print(f"Average test set BLEU score: {test_score}")


    def train_loop(self, train_loader, val_loader):

        epochs = self.config["max_epochs"]
        batch_size = self.config["batch_size"]

        train_losses = []
        val_losses = []

        best_val_loss = float("inf")
        best_loss_epoch = None

        for epoch in range(1, epochs+1):
            self.model.train()
            print(f"Training epoch {epoch}/{epochs}")
            train_loss = self.train_step(train_loader, epoch)
            self.model.eval()
            print(f"Validation epoch {epoch}/{epochs}")
            val_loss = self.val_step(val_loader, epoch)

            if val_loss < best_val_loss:
                if best_loss_epoch != None:
                    os.system(f"rm {CHECKPOINT_DIR}/model_{self.model_name}_{best_loss_epoch}_checkpoint.pt")
                best_val_loss = val_loss
                best_loss_epoch = epoch
                torch.save(self.model.state_dict(), f"{CHECKPOINT_DIR}/model_{self.model_name}_{epoch}_checkpoint.pt")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch} train loss: {train_loss}, val_loss: {val_loss}")

        self.best_epoch = best_loss_epoch

        self.generate_learning_curvers(train_losses, val_losses)
        


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

                logits = output.logits

                print(logits.shape)
                print(target_ids.shape)

                logits_dim = logits.shape[-1]

                logits = logits[1:].view(-1, logits_dim)
                target_ids = target_ids[1:].reshape(-1)

                print(logits.shape)
                print(target_ids.shape)

                loss = self.criterion(logits, target_ids)
                
                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

                if (step+1) % 10 == 0:
                    print(f"\nEpoch {epoch}, samples {step+1}/{n} train loss: {total_loss/(step+1)}")

                pbar.update(1)
                
                break

                
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
                logits = output.logits

                logits_dim = logits.shape[-1]

                logits = logits[1:].view(-1, logits_dim)
                target_ids = target_ids[1:].reshape(-1)

                loss = self.criterion(logits, target_ids)

                total_loss += loss.item()

                if (step+1) % 10 == 0:
                    print(f"\nEpoch {epoch}, samples {step+1}/{n} train loss: {total_loss/(step+1)}")

                pbar.update(1)

                break

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
                logits = output.logits

                logits_dim = logits.shape[-1]

                logits = logits[1:].view(-1, logits_dim)
                target_ids = target_ids[1:].reshape(-1)

                loss = self.criterion(logits, target_ids)

                total_loss += loss.item()

                pbar.update(1)

                break

        avg_loss = total_loss / n

        return avg_loss
            

    

    def metric_evaluation(self, data_loader, generate_fun):
        
        self.model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/model_{self.model_name}_{self.best_epoch}_checkpoint.pt"))
        self.model.eval()


        score = 0

        for step, batch in enumerate(data_loader):

            self.optimizer.zero_grad()

            inputs, targets = batch

            for i in range(len(inputs.input_ids)):

                input_ids = inputs.input_ids[i]
                target_ids = targets.input_ids[i]

                print("AAAAAAAAAAAAAAAAAAAAAAAAAA")
                output = generate_fun(input_ids.unsqueeze(0))
                print("AAAAAAAAAAAAAAAAAAAAAAAAAA")

                if type(output) == tuple:
                    pred_ids, attention = output
                else:
                    pred_ids = output[0]
                # source_tokens = self.src_tokenizer.convert_ids_to_tokens(pred_ids)
                # target_tokens = self.dst_tokenizer.convert_ids_to_tokens(target_ids)

                pred_sentence = self.src_tokenizer.decode(pred_ids, skip_special_tokens=True)
                target_sentence = self.dst_tokenizer.decode(target_ids, skip_special_tokens=True)

                print(pred_sentence, target_sentence)

                result = self.metric.compute(predictions=[pred_sentence], references=[target_sentence])
                score += result["bleu"]

                break

            score /= len(data_loader)

            return score