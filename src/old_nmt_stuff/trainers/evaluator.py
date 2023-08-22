from sys import path
path.append("./trainers")
from trainer_constants import *
import torch
from torch import nn
from torch import optim
from trainer import Trainer
from tqdm import tqdm

class Evaluator(Trainer):


    def __init__(self, model, src_tokenizer, dst_tokenizer, config) -> None:
        super().__init__(model, src_tokenizer, dst_tokenizer, config)

    
    def train(self, generate_fun):
        
        seed = self.config["seed"]
        self.set_seeds(seed)

        batch_size = self.config["batch_size"]
        train_loader, val_loader, test_loader = self.get_data_loader(batch_size, 0.2, 0.1)
        self.model.eval()

        train_score = self.metric_evaluation(train_loader, generate_fun)
        val_score = self.metric_evaluation(val_loader, generate_fun)
        test_score = self.metric_evaluation(test_loader, generate_fun)

        print(f"Average train set BLEU score: {train_score}")
        print(f"Average validation set BLEU score: {val_score}")
        print(f"Average test set BLEU score: {test_score}")