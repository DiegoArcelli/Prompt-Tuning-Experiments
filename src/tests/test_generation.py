from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5Tokenizer, T5ForConditionalGeneration
import torch
import sys
sys.path.append("./../")
from models.nmt_models import T5ForNMT
from models.prompt_tuning_models import T5PromptTuning


model = T5ForConditionalGeneration.from_pretrained("t5-small")
# model = T5ForNMT(1024, 32000)

# model = T5PromptTuning.from_pretrained(
#     "t5-small",
#     encoder_soft_prompt_path = None,
#     decoder_soft_prompt_path = None,
#     encoder_n_tokens = 20,
#     decoder_n_tokens = 20,
#     encoder_hidden_dim=64,
#     decoder_hidden_dim=64
# )

tokenizer = T5Tokenizer.from_pretrained("t5-small")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)


model.eval()

prompt = "translate German to English: Würde ich wenn ich gut darin wäre"

# Tokenize prompt
encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

# Generate text
# output_sequences = model.generate(
#     input_ids=encoded_prompt,
#     max_length=50,
#     temperature=1.0,
#     top_k=0,
#     top_p=0.9,
#     do_sample=True,
#     num_return_sequences=1,
# )


# out = model(input_ids=encoded_prompt)
# print(out.keys())

output_sequences = model.generate(
    input_ids=encoded_prompt, 
    decoder_input_ids=torch.zeros([1,1]).long(), 
    max_length=10,
    num_beams=3,
    do_sample=True,
    no_repeat_ngram_size=1,  
    temperature = 1.0,
    top_k = 0,
    top_p = 0.8,
    repetition_penalty = 1.0,
    use_cache=False,
    early_stopping=True,
)

# # Decode generated text
generated_text = tokenizer.decode(output_sequences[0], clean_up_tokenization_spaces=True)
print(generated_text)