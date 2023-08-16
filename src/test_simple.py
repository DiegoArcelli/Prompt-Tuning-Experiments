import sys
sys.path.append("./models/")
from models.prompt_tuning_models_multiple import MT5PromptTuningSimple
from utils import count_parameters
from transformers import AutoTokenizer, MT5ForConditionalGeneration
import torch

model = MT5PromptTuningSimple.from_pretrained("google/mt5-small", None, None, 10, 10)

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

prefix = "translate English to German"

# prompt = "translate German to English: Würde ich wenn ich gut darin wäre"
prompt = [f"{prefix}: I don't speak german.", f"{prefix}: You are really bad.", f"{prefix}: Blue moon you are no longer alone"]

# Tokenize prompt
encoded_prompt = tokenizer(prompt, max_length=20, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
print(encoded_prompt)
output_sequences = model.generate(
    **encoded_prompt,
    max_length=10,
    # decoder_input_ids=torch.zeros([1,1]).long(), 
    # max_length=200,
    # num_beams=5
    # early_stopping=True,
)
# # Decode generated text
generated_text = tokenizer.decode(output_sequences[0], clean_up_tokenization_spaces=True)
print(generated_text)

# for n, p in model.named_parameters():
#     if p.requires_grad:
#          print(n, p.shape)