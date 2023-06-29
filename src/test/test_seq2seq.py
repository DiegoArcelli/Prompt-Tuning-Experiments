import sys
sys.path.append("./../")
import torch
from torch import nn
import torch.functional as F
from models.rnn_models import Seq2Seq, Decoder, Encoder, AttentionLayer
from transformers import AutoTokenizer
from utils import plot_attention_mask
import evaluate

bleu = evaluate.load("bleu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 5000
src_len = 15
dst_len = 10
batch_size = 64
src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
dst_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# dst = ["I love you", "Please don't do it"]
# src = ["Ti amo", "Per favore non lo fare"]


# x = src_tokenizer(src, max_length=src_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
# x = x.input_ids.permute(1, 0)
# y = dst_tokenizer(dst, max_length=dst_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
# y = y.input_ids.permute(1, 0)

model = Seq2Seq(
    enc_vocab_dim=32128,
    dec_vocab_dim=32128,
    enc_hidden_dim=4,
    dec_hidden_dim=4,
    enc_n_layers=2,
    dec_n_layers=1,
    pad_idx=src_tokenizer.pad_token_id,
    start_idx=dst_tokenizer.sep_token_id,
    end_idx=dst_tokenizer.mask_token_id,
    teacher_forcing_ratio=0.5,
    device=device
)


# print(x.shape)
# print(y.shape)

# out = model(x, y)
# print(out.shape)

prompt = "I like bikes"

emb_prompt = src_tokenizer([prompt], max_length=src_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
pred, attention_mask = model.generate(emb_prompt.input_ids, 10)

print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa")

source_tokens = src_tokenizer.convert_ids_to_tokens(emb_prompt.input_ids[0])
target_tokens = dst_tokenizer.convert_ids_to_tokens(pred)

plot_attention_mask(attention_mask.to("cpu"), source_tokens, target_tokens)

source_sentence = src_tokenizer.decode(emb_prompt.input_ids[0], skip_special_tokens=True)
target_sentence = dst_tokenizer.decode(pred, skip_special_tokens=True)

print(source_sentence)
print(target_sentence)

results = bleu.compute(predictions=[target_sentence], references=[source_sentence])
print(results["bleu"])