from transformers import AutoTokenizer

with open("./../dataset/ita.txt", "r") as dataset:
    sentences = [sentence.split("\t")[:2] for sentence in dataset.readlines()]

sentences_src = [x for [x, _] in sentences]
sentences_dst = [x for [_, x] in sentences]


# src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
# dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
src_tokenizer = AutoTokenizer.from_pretrained("t5-small")
dst_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")

src = max([len(src_tokenizer(f"translate English to Italian: {x}").input_ids) for x in sentences_src])
dst = max([len(dst_tokenizer(x).input_ids) for x in sentences_dst])

print("Max length source: ", src)
print("Max length destination: ", dst)

print("Source tokenizer vocabulary size: ", src_tokenizer.vocab_size)
print("Destination tokenizer vocabulary size: ", dst_tokenizer.vocab_size)