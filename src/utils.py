def count_parameters(model):
    n_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {n_params} trainable parameters')