from torch.utils.data import Dataset
import random
from tqdm import tqdm

class AnkiDataset(Dataset):

    def __init__(self,
                 data_path,
                 tokenizer_src,
                 tokenizer_dst,
                 src_max_length,
                 dst_max_length,
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
        random.seed(self.seed)
        data = self.get_data(data_path)
        self.data = self.get_data(data)
        self.prefix = prefix
        self.lang = lang


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        
        src, dst = self.data[index]
        return (src, dst)
        

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

        tokenized_data = []
        print("Tokenizing the dataset")
        with tqdm(total=len(data)) as pbar:
            for (src, dst) in self.data:

                src = f"translate English to {language}: {src}" if self.prefix else src
                src = self.tokenizer_src(src, max_length=self.src_max_length, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                dst = self.tokenizer_dst(dst, max_length=self.dst_max_length, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
                
                for key in src.keys():
                    src[key] = src[key][0]
                    dst[key] = dst[key][0]

                tokenized_data.append((src, dst))
                pbar.update(1)

        return tokenized_data