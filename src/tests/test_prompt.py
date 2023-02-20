import sys
sys.path.append("./../")
from transformers import T5Tokenizer
from models.prompt_tuning_models import T5PromptTuning


model = T5PromptTuning.from_pretrained(
    "t5-small",
    encoder_soft_prompt_path = None,
    decoder_soft_prompt_path = None,
    encoder_n_tokens = 20,
    decoder_n_tokens = 20,
)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

inputs = tokenizer(["Hello, my dog is cute", "I hate black cats", "Stop it"], padding=True, truncation=True, return_tensors="pt")
targets = tokenizer(["Ciao, il mio cane è carino", "Odio i gatti neri", "Smettila"], padding=True, truncation=True, return_tensors="pt")


# print(inputs.attention_mask, inputs.attention_mask.shape)
# print(targets.attention_mask, targets.attention_mask.shape)

print("\n\n")


out = model(
    input_ids = inputs.input_ids,
    attention_mask=inputs.attention_mask,
    decoder_input_ids=targets.input_ids,
    decoder_attention_mask=targets.attention_mask
)
print(out.keys())