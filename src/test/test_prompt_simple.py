import sys
sys.path.append("./../")
from models.prompt_tuning_models import T5PromptTuningSimple
from transformers import T5Tokenizer

model = T5PromptTuningSimple.from_pretrained(
    "t5-small",
    encoder_soft_prompt_path = None,
    decoder_soft_prompt_path = None,
    encoder_n_tokens = 40,
    decoder_n_tokens = 40,
)

tokenizer = T5Tokenizer.from_pretrained("t5-small")

inputs = tokenizer(["Hello, my dog is cute", "I hate black cats"], padding=True, truncation=True, return_tensors="pt")
targets = tokenizer(["Ciao, il mio cane è carino", "Odio i gatti neri"], padding=True, truncation=True, return_tensors="pt")

print(inputs.input_ids.shape, targets.input_ids.shape)

out = model(input_ids=inputs.input_ids, decoder_input_ids = targets.input_ids)
print(out.keys())
logits = out.logits
print(logits.shape)
