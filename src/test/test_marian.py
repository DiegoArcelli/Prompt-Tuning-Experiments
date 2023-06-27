from transformers import MarianMTModel, MarianTokenizer

src_text = [
    "I know Tom wants to talk with Mary",
    "Nobody goes there anymore",
    "I still care",
    "Are you going?"
]

model_name = "Helsinki-NLP/opus-mt-tc-big-en-it"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
tok = tokenizer(src_text, return_tensors="pt", padding=True)

print(tok)

translated = model.generate(**tok)

for t in translated:
    print(tokenizer.decode(t, skip_special_tokens=True))