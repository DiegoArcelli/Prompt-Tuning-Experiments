from transformers import BartModel, T5Model

def count_parameters(model):
    n_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {n_params} trainable parameters')


def get_model( model_name):

    assert model_name in ["bart", "t5"]

    match model_name:
        case "bart":
            model = BartModel.from_pretrained("facebook/bart-base")
        case "t5":
            model = T5Model.from_pretrained("t5-small")

    return model