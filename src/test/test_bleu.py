import evaluate

bleu = evaluate.load("bleu")
sacre_bleu = evaluate.load("sacrebleu")


predictions = ["hello there general kenobi", "foo bar foobar"]

references = [
    ["hello there general kenobi", "hello there !"],
    ["foo bar foobar"]
]



results = bleu.compute(predictions=predictions, references=references)

print(results)

# results = sacre_bleu.compute(predictions=predictions, references=references)

# print(results)