import sys
sys.path.append("./../")
from models.prompt_tuning_models import T5PromptTuningSimple


model = T5PromptTuningSimple.from_pretrained(
    "t5-small",
    encoder_soft_prompt_path = None,
    decoder_soft_prompt_path = None,
    encoder_n_tokens = 40,
    decoder_n_tokens = 40,
    encoder_hidden_dim=32,
    decoder_hidden_dim=32
)

print(model)