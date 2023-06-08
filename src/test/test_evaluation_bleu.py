import sys
sys.path.append("./../")
sys.path.append("./../trainer/")
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5Tokenizer, T5ForConditionalGeneration
from nmt_datasets import AnkiDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from models.nmt_models import T5ForNMT
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.prompt_tuning_models import T5PromptTuning
import time
import evaluate

device = "cpu"


metric = evaluate.load("bleu")

def metric_evaluation(model, data_loader, generate_fun):
        
    model.eval()


    score = 0

    for step, batch in enumerate(data_loader):

        inputs, targets = batch

        inputs = inputs.to(device)
        targets = targets.to(device)

        for i in range(len(inputs.input_ids)):

            input_ids = inputs.input_ids[i]
            target_ids = targets.input_ids[i]

            output = generate_fun(input_ids.unsqueeze(0))

            if type(output) == tuple:
                pred_ids, attention = output
            else:
                pred_ids = output[0]

            pred_sentence = tokenizer.decode(pred_ids, skip_special_tokens=True)
            target_sentence = tokenizer.decode(target_ids, skip_special_tokens=True)

            result = metric.compute(predictions=[pred_sentence], references=[target_sentence])

            if result["bleu"] > 0:
                print(f"Target: {target_sentence}")
                print(f"Predicted: {pred_sentence}")
                print(f"BLEU: {result['bleu']}\n")
                #print(pred_sentence, target_sentence, result["bleu"])

            score += result["bleu"]

        score /= len(data_loader)

        return score


model_name = "SEBIS/legal_t5_small_trans_en_it"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

val_split = 0.2
test_split = 0.1 

data_set = AnkiDataset(
    f"./../../dataset/ita.txt",
    tokenizer,
    tokenizer,
    200,
    200
)


n = len(data_set)

val_size = int(n*val_split)
test_size = int(n*test_split)
train_size = n - val_size - test_size

batch_size = 32

train_set, val_set, test_set = random_split(data_set, [train_size, val_size, test_size])

train_loader = DataLoader(
    train_set,
    batch_size = batch_size
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size
)

test_loader = DataLoader(
    test_set,
    batch_size = batch_size
)

generate_fun = lambda x: model.generate(
    input_ids=x, 
    decoder_input_ids=torch.zeros([1,1]).long().to(device), 
    max_length=200,
    num_beams=5,
    early_stopping=True,
)


score = metric_evaluation(model, test_loader, generate_fun)
print("\n=========")
print(score)
print("=========")