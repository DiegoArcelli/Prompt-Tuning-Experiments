from transformers import BartModel, T5Model, T5ForConditionalGeneration
from transformers import AutoTokenizer

ita_sentence = "Amo il mio cane"
eng_sentence = "I love my dog"

src_text = [
    "I love my dog",
    "I hate that thing"
]

dst_text = [
    "Amo il mio cane",
    "Odio quella cosa"
]

src_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
dst_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")


# inputs = src_tokenizer(eng_sentence, return_tensors='pt', padding=True, truncation=True)
# targets = dst_tokenizer(ita_sentence, return_tensors='pt', padding=True, truncation=True)
inputs = src_tokenizer(
            src_text,
            max_length=10,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )


targets = dst_tokenizer(
            dst_text,
            max_length=10,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )


print(inputs)
print(src_tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))

# model = T5Model.from_pretrained('t5-base')
# model = T5ForConditionalGeneration.from_pretrained("t5-base")
model = T5Model.from_pretrained("t5-base")
model.train()

# y_ids = y[:, :-1].contiguous()
y = targets.input_ids
lm_labels = y[:, 1:].clone().detach()
lm_labels[y[:, 1:] == dst_tokenizer.pad_token_id] = -100

output = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    decoder_input_ids=targets.input_ids,
    labels=y
)

print(output.keys())
# print(output.logits.shape)
print(output.encoder_last_hidden_state.shape)