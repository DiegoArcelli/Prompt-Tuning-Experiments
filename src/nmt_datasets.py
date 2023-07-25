import random
from tqdm import tqdm
from torch.utils.data import random_split
from datasets import Dataset, DatasetDict


class AnkiDatasetFactory:

    def __init__(self,
                 data_path,
                 tokenizer_src,
                 tokenizer_dst,
                 src_max_length,
                 dst_max_length,
                 test_split=0.1,
                 val_split=0.1,
                 prefix=False,
                 subsample=False,
                 frac=1.0,
                 seed=42,
                 lang="ita"
                ) -> None:
        super().__init__()

        self.tokenizer_src = tokenizer_src
        self.tokenizer_dst = tokenizer_dst
        self.src_max_length = src_max_length
        self.dst_max_length = dst_max_length
        self.seed = seed
        self.frac = frac
        self.subsample = subsample
        self.prefix = prefix
        self.lang = lang

        random.seed(self.seed)
        data = self.get_data(data_path)
        data = self.tokenize_data(data)
        data = Dataset.from_list(data)


        train_test_data = data.train_test_split(test_size=test_split, seed=self.seed)
        train_val_data = train_test_data["train"].train_test_split(test_size = val_split, seed=self.seed)
        del train_test_data["train"]
        self.train_val_data = train_val_data
        self.test_data = train_test_data
    '''
    Takes in input the path of the datasets and it returnes a list where each element of
    the list is a list of the elment containing the english and italian sentence
    '''
    def get_data(self, data_path="./../dataset/ita.txt"):

        with open(data_path, "r") as dataset:
            sentences = [tuple(sentence.split("\t")[:2]) for sentence in dataset.readlines()]
            
        if self.subsample == True:
            k = int(len(sentences)*self.frac)
            sentences = random.sample(sentences, k)

        return sentences
    

    def tokenize_data(self, data):

        if self.lang == "ita":
            language = "Italian"
        elif self.lang == "deu":
            language = "German"

        records = []
        print("Tokenizing the dataset")
        with tqdm(total=len(data)) as pbar:
            for idx, (src, dst) in enumerate(data):
                record = {}

                pre_src = f"translate English to {language}: {src}" if self.prefix else src
                src_tokenized = self.tokenizer_src(pre_src, max_length=self.src_max_length, truncation=True, return_tensors='pt')
                dst_tokenized = self.tokenizer_dst(dst, max_length=self.dst_max_length, truncation=True, return_tensors='pt')
                
                for key in src_tokenized.keys():
                    src_tokenized[key] = src_tokenized[key][0]
                    dst_tokenized[key] = dst_tokenized[key][0]

                record["id"] = idx
                record["translation"] = {
                    'en': src,
                    'it': dst
                }
                record["input_ids"] = src_tokenized.input_ids
                record["attention_mask"] = src_tokenized.attention_mask
                record["labels"] = dst_tokenized.input_ids


                records.append(record)
                pbar.update(1)

        return records