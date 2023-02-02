from torch.utils.data import Dataset, RandomSampler
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import time

'''
Takes in input the path of the datasets and it returnes a list where each element of
the list is a list of the elment containing the english and italian sentence
'''
def get_data(data_path="./../dataset/ita.txt"):

    with open(data_path, "r") as dataset:
        sentences = dataset.readlines()

    sentences = [tuple(sentence.split("\t")[:2]) for sentence in sentences]

    return sentences

class AnkiDataset(Dataset):

    def __init__(self, data_path, tokenizer_src, tokenizer_dst) -> None:
        super().__init__()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_dst = tokenizer_dst
        sentences = get_data(data_path)
        self.data = sentences
        

    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, index):
        return self.data[index]


EPOCHS = 1
BATCH_SIZE = 64


model_name = "dbmdz/bert-base-italian-cased"

src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

train_dataset = AnkiDataset("./../dataset/ita.txt", src_tokenizer, dst_tokenizer)

train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            batch_size = BATCH_SIZE # Trains with this batch size.
        )


for epoch in range(1, EPOCHS+1):

    for step, batch in enumerate(train_dataloader):
        inputs, targets = batch
        inputs = src_tokenizer(list(inputs))
        targets = dst_tokenizer(list(targets))