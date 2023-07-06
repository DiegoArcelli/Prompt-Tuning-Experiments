from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("gsarti/it5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("gsarti/it5-small")

generate_fun = lambda x: model.generate(
    **x, 
    #decoder_input_ids=torch.zeros([1,1]).long().to(device), 
    # max_length=200,
    # early_stopping=True,
)

prefix = "translate Italian to English"
test_sentences = ["Ciao come stai? Io bene "]
# test_sentences = [f"{prefix}: {sentence}" for sentence in test_sentences]
inputs = tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt")
outputs = generate_fun(inputs)
generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(generated)
# print(model)
