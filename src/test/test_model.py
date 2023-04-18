from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
import torch
import sys
sys.path.append("./../")
from models.nmt_models import T5ForNMT
from models.prompt_tuning_models import T5PromptTuning


model = T5ForConditionalGeneration.from_pretrained("t5-small")
# model = T5ForNMT.from_pretrained("t5-small", hidden_size=512, voc_size=32128)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model.eval()

input_sentence = "hello how are you"
target_sentence = "ciao come stai"
enc_input = tokenizer(input_sentence, add_special_tokens=False, return_tensors="pt")
enc_target = tokenizer(target_sentence, add_special_tokens=False, return_tensors="pt")

output = model.forward(
    input_ids=enc_input.input_ids,
    decoder_input_ids=enc_target.input_ids,
    output_attentions=True
)

print(output.keys())

# print(output.keys())
# # print(output["last_hidden_state"].shape)
# print(output["logits"].shape)

# print(encoded_prompt.shape)

# output_sequences = model.generate(
#     input_ids=encoded_prompt.input_ids, 
#     decoder_input_ids=torch.zeros([1,1]).long(),
#     output_attentions=True,
#     max_length=10,
#     num_beams=3,
#     no_repeat_ngram_size=1,  
#     early_stopping=True,
# )

# print(output_sequences)

# generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
# print(generated_text)