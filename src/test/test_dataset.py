import sys
sys.path.append("./../")
from nmt_datasets import AnkiDataset
from transformers import AutoTokenizer

config = {
    "src_max_length": 183,
    "dst_max_length": 208,
    "src_vocab_size": 31102,
    "dst_vocab_size": 28996,
    "enc_hidden_dim": 8,
    "dec_hidden_dim": 8,
    "max_epochs": 1,
    "batch_size": 16,
    "seed": 7,
}


src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')



data_set = AnkiDataset(
            f"../../dataset/ita.txt",
            src_tokenizer,
            dst_tokenizer,
            config["src_max_length"],
            config["dst_max_length"],
            subsample=True,
            frac=0.1
        )

print(len(data_set))