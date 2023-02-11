from transformers import AutoTokenizer

with open("./../dataset/ita.txt", "r") as dataset:
    sentences = [sentence.split("\t")[:2] for sentence in dataset.readlines()]

sentences_src = [x for [x, _] in sentences]
sentences_dst = [x for [_, x] in sentences]


src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

src = max([len(src_tokenizer(x).input_ids) for x in sentences_src])
dst = max([len(dst_tokenizer(x).input_ids) for x in sentences_dst])

print("Max length source: ", src)
print("Max length destination: ", dst)