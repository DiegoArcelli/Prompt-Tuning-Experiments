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
from transformers import MarianMTModel, MarianTokenizer
from models.prompt_tuning_models import T5PromptTuning
import time
from tqdm import tqdm
import evaluate

device = "cpu"


metric = evaluate.load("bleu")

def metric_evaluation(model, data_loader, generate_fun):
        
    model.eval()
    score = 0

    n = 0

    # prefix = "translate English to Italian:"
    # prefix_tok = tokenizer(prefix, pad_to_max_length=True, truncation=True, return_tensors='pt')


    with tqdm(total=len(data_loader)) as pbar:
        for step, batch in enumerate(data_loader):

            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            # output = model.generate(**inputs)
            output = generate_fun(inputs)
            pred_sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
            org_sentences = tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)
            target_sentences = tokenizer.batch_decode(targets.input_ids, skip_special_tokens=True)
            # result = metric.compute(predictions=pred_sentences, references=target_sentences)
            

            for i in range(len(pred_sentences)):
                pred = pred_sentences[i].replace("▁", " ")
                targ = target_sentences[i].replace("▁", " ")
                org = org_sentences[i].replace("▁", " ")
                result = metric.compute(predictions=[pred], references=[targ])
                score += result["bleu"]
                print(org, pred, "\n\n")
                n+=1


            pbar.update(1)

    score /= n
    return score


# model_name = "SEBIS/legal_t5_small_trans_en_it_small_finetuned"
# model_name = "t5-base"
model_name = "Helsinki-NLP/opus-mt-tc-big-en-it"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

val_split = 0.2
test_split = 0.1 

data_set = AnkiDataset(
    f"./../../dataset/ita.txt",
    tokenizer,
    tokenizer,
    200,
    200,
    subsample=True,
    frac=0.005,
    prefix=False
)


n = len(data_set)
print(n)

val_size = int(n*val_split)
test_size = int(n*test_split)
train_size = n - val_size - test_size

batch_size = 4

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
    **x, 
    #decoder_input_ids=torch.zeros([1,1]).long().to(device), 
    # max_length=200,
    # early_stopping=True,
)

print(len(test_loader))
score = metric_evaluation(model, test_loader, generate_fun)
print("\n=========")
print(score)
print("=========")