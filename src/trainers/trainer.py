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

class Trainer:

    def __init__(self, model, src_tokenizer, dst_tokenizer, config) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.dst_tokenizer = dst_tokenizer
        self.config = config

        pad_token = dst_tokenizer.pad_token
        pad_token_idx = dst_tokenizer.convert_tokens_to_ids([pad_token])[0]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)

        if "model_name" in config:
            self.model_name = config["model_name"]
        else:
            self.model_name = self.model.__class__.__name__.lower()


    
    def set_seeds(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


    def get_data_loader(self, batch_size, val_split=0.2, test_split=0.1):
        data_set = AnkiDataset(f"{DATASET_PATH}/ita.txt",
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


    def train(self):
        
        seed = self.config["seed"]
        self.set_seeds(seed)

        batch_size = self.config["batch_size"]
        # self.model.to(self.device)

        train_loader, val_loader, test_loader = self.get_data_loader(batch_size, 0.2, 0.1)

        self.train_loop(train_loader, val_loader)
        self.model.eval()
        self.test_step(train_loader, val_loader, test_loader)




    def train_loop(self, train_loader, val_loader):

        epochs = self.config["max_epochs"]
        batch_size = self.config["batch_size"]

        train_losses = []
        val_losses = []

        best_val_loss = float("inf")
        best_loss_epoch = None

        for epoch in range(1, epochs+1):
            self.model.train()
            train_loss = self.train_step(train_loader)
            self.model.eval()
            val_loss = self.val_step(val_loader)

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
        


    def train_step(self, train_loader):

        for step, batch in enumerate(train_loader):

            self.optimizer.zero_grad()

            inputs, targets = batch

            output = self.model(inputs, targets)

            logits = output.logits

    
    def val_step(self, val_loader):

        for step, batch in enumerate(val_loader):

            inputs, targets = batch

            output = self.model(inputs, targets)

            logits = output.logits


    
    def test_step(self, train_loader, val_loader, test_loader):

        self.model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/model_{self.model_name}_{self.best_epoch}_checkpoint.pt"))
        
        for step, batch in enumerate(train_loader):

            inputs, targets = batch
            
            output = self.model(inputs, targets)

            logits = output.logits