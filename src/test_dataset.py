from nmt_datasets import AnkiDatasetFactory
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

data_set = AnkiDatasetFactory(
            f"../dataset/ita.txt",
            src_tokenizer,
            dst_tokenizer,
            config["src_max_length"],
            config["dst_max_length"],
            subsample=True,
            frac=0.01,
            seed=7,
            lang="ita"
        )

print(data_set.train_val_data)
print(data_set.test_data)

# trainer = Trainer(model, src_tokenizer, dst_tokenizer, config)
# trainer.train()

