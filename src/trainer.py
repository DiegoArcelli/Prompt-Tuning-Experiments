from torch.utils.data import Dataset, RandomSampler
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import BartModel
from models import BartForNMT
import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split
import random
import numpy as np
import time

seed = 7

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class AnkiDataset(Dataset):

    def __init__(self, data_path, tokenizer_src, tokenizer_dst) -> None:
        super().__init__()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_dst = tokenizer_dst
        self.data = self.get_data(data_path)

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, index):

        sentence_src = self.data[0][index]
        sentence_dst = self.data[1][index]

        src = self.tokenizer_src.batch_encode_plus([sentence_src], max_length=512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
        dst = self.tokenizer_dst.batch_encode_plus([sentence_dst], max_length=512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

        return src, dst
    

    '''
    Takes in input the path of the datasets and it returnes a list where each element of
    the list is a list of the elment containing the english and italian sentence
    '''
    def get_data(self, data_path="./../dataset/ita.txt"):
        with open(data_path, "r") as dataset:
            sentences = [sentence.split("\t")[:2] for sentence in dataset.readlines()]

        org = [x for [x, _] in sentences]
        dst = [x for [_, x] in sentences]

        return (org, dst)


class Trainer:

    def __init__(self, model) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model


    def train(self):

        # self.model.to(self.device)

        EPOCHS = 1
        BATCH_SIZE = 8

        src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
        dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

        data_set = AnkiDataset("./../dataset/ita.txt", src_tokenizer, dst_tokenizer)


        train_size = int(len(data_set)*0.8)
        test_size = len(data_set) - train_size
        train_set, test_set = random_split(data_set, [train_size, test_size])

        train_loader = DataLoader(
                    train_set,
                    batch_size = BATCH_SIZE
                )
        
        test_loader = DataLoader(
                    test_set,
                    batch_size = BATCH_SIZE
                )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)


        for epoch in range(1, EPOCHS+1):

            self.model.train()

            for step, batch in enumerate(train_loader):

                optimizer.zero_grad()

                inputs, targets = batch

                print(batch)

                return

                # outputs = model(inputs)
                # print(outputs.keys())

                # logits = outputs.logits

                # print(logits.shape)

                # # print(logits.view(-1, 50265).shape)
                # # print(targets.view(-1).shape)

                # exit()

            



# model = BartModel.from_pretrained('facebook/bart-base')
model = BartForNMT(768, 50265)
trainer = Trainer(model)
trainer.train()