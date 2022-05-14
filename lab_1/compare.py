import torch
import matplotlib.pyplot as plt

def get_comparison(model_1, model_2, data, title, label1, label2):
    with torch.no_grad():
        scores, activations = model_1(data)

    res = {}
    for i, layer in enumerate(activations):
        layer = layer.to(device='cpu')
        summarize = torch.zeros(layer.shape[1] * layer.shape[2] * layer.shape[3])
        for activation in layer:
            summarize += torch.ravel(activation)
        res[i] = summarize / 64
    res1 = res

    with torch.no_grad():
        scores, activations = model_2(data)

    res = {}
    for i, layer in enumerate(activations):
        layer = layer.to(device='cpu')
        summarize = torch.zeros(layer.shape[1] * layer.shape[2] * layer.shape[3])
        for activation in layer:
            summarize += torch.ravel(activation)
        res[i] = summarize / 64
    res2 = res

    fig, axs = plt.subplots(2, len(activations), sharex=True, sharey=True, figsize=(12, 6))
    fig.suptitle(title, fontsize=14)
    for i in range(len(activations)):
        axs[0, i].set_title(f'Слой {i + 1}')
        axs[0, i].grid()
        axs[0, 0].set_ylabel(label1)
        axs[1, i].set_xlabel('активация')
        axs[1, 0].set_ylabel(label2)
        axs[1, i].grid()

        a = res1[i][res1[i] < 2]
        axs[0, i].hist(a, bins=20)
        axs[0, i].set_xticks([0, 0.5, 1, 1.5, 2])

        a = res2[i][res2[i] < 2]
        axs[1, i].hist(a, bins=20)
        axs[1, i].set_xticks([0, 0.5, 1, 1.5, 2])
    fig.show()