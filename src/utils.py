from transformers import BartModel, T5Model
import matplotlib.pyplot as plt


def count_parameters(model):
    n_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {n_params} trainable parameters')


def get_model(model_name):

    assert model_name in ["bart", "t5"]

    match model_name:
        case "bart":
            model = BartModel.from_pretrained("facebook/bart-base")
        case "t5":
            model = T5Model.from_pretrained("t5-small")

    return model


def plot_curves(curve_1, label_1, curve_2=None, label_2=None, fig_name="figure", show=False):
    plt.plot(curve_1, label = label_1)
    if curve_2 is not None:
        plt.plot(curve_2, label = label_2)
    plt.legend()
    plt.savefig(f"{fig_name}")

    if show:
        plt.show()

    