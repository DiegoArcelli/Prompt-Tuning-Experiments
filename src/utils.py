import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType, PromptTuningConfig, PromptEncoderConfig
from peft import PromptEncoderConfig, PromptEncoder


def load_model(model_type="generation", model_name="t5-small", mode="normal", num_tokens=20):

    if mode == "normal":
        if model_type == "generation":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif model_type == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    elif mode == "prompt":
        if model_type == "generation":
            peft_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=num_tokens)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif model_type == "sequence_classification":
            peft_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=20)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=2)
            model = get_peft_model(model, peft_config)
            for n, p in model.named_parameters():
                if "classifier" in n:
                    p.requires_grad = True
            model.print_trainable_parameters()

    elif mode == "prefix":
        if model_type == "generation":
            peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=num_tokens)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif model_type == "sequence_classification":
            peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=20)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=2)
            model = get_peft_model(model, peft_config)
            for n, p in model.named_parameters():
                if "classifier" in n:
                    p.requires_grad = True
            model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def count_parameters(model):
    n_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {n_params} trainable parameters')


def plot_curves(curve_1, label_1, curve_2=None, label_2=None, fig_name="figure", show=False):

    plt.plot(curve_1, label = label_1)
    if curve_2 is not None:
        plt.plot(curve_2, label = label_2)
    plt.legend()
    plt.savefig(f"{fig_name}")

    if show:
        plt.show()

    plt.clf()

    
def plot_attention_mask(attention_mask, source_tokens, target_tokens):

    skip_tokens = len(source_tokens) if "[PAD]" not in source_tokens else source_tokens.index("[PAD]")
    source_tokens = source_tokens[:skip_tokens]

    attention_mask = attention_mask.squeeze(1)

    attention_mask = attention_mask[:, :skip_tokens]

    plt.xticks(ticks=[x for x in range(len(source_tokens))], labels=source_tokens, rotation=45)
    plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.yticks(ticks=[x for x in range(len(target_tokens))], labels=target_tokens)
    plt.imshow(attention_mask, cmap='gray', vmin=0, vmax=1)
    plt.show()
    


def test_generation(model, src_tokenizer, dst_tokenizer, prefix = None, test_sentences = None):
    if test_sentences == None:
        test_sentences = ["I like pizza.", "I love my dog.", "How old are you?"]
    if prefix is not None:
        test_sentences = [f"{prefix}: {sentence}" for sentence in test_sentences]
    inputs = src_tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)

    gen_inputs = dict()
    gen_inputs["input_ids"] = inputs["input_ids"]
    gen_inputs["attention_mask"] = inputs["attention_mask"]

    generated = model.generate(**gen_inputs)
    return dst_tokenizer.batch_decode(generated, skip_special_tokens=True)