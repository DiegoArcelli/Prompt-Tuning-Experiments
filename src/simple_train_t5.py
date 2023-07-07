from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import Seq2SeqTrainer

model = T5ForConditionalGeneration.from_pretained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

