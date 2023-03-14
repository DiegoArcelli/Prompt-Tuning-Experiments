import matplotlib.pyplot as plt


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

    
def print_attention_mask(attention_mask):
    attention_mask = attention_mask.squeeze(1)
    plt.imshow(attention_mask)
    plt.show()
    
