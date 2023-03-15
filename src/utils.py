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
    
