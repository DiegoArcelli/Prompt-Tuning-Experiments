from transformers import AutoTokenizer
import re
from functools import reduce

# read dataset from file system
with open("../dataset/ita.txt", "r") as dataset:
    sentences = dataset.readlines()


# remove setence from text
sentences = [sentence.split("\t")[:2] for sentence in sentences]

src_max_length = max([len(sentence[0]) for sentence in sentences])
dst_max_length = max([len(sentence[1]) for sentence in sentences])
src_min_length = min([len(sentence[0]) for sentence in sentences])
dst_min_length = min([len(sentence[1]) for sentence in sentences])

print(f"Number of sentences in the dataset {len(sentences)}")
print(f"Max length of an English sentence {src_max_length}\nMax length of an Italian sentence {dst_max_length}")
print(f"Min length of an English sentence {src_min_length}\nMin length of an Italian sentence {dst_min_length}")


n = 10
print("Some examples:")
[print(x) for x in sentences[:n]]

regex = re.compile('[^a-zA-Z]')

joined_pairs = list(map(lambda x: x[0] + " " + x[1], sentences))
merged_text = "\n".join(joined_pairs)

non_chars = set(re.findall(regex, merged_text))
print(f"Characters which are not letters present in the dataset:\n{non_chars}")
