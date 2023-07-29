import torch
from torch import nn


def visualize_outputs(model, z_dim, device):
    model.eval()

    with torch.no_grad():
        noise = torch.randn(10, z_dim, 1, 1).to(device)
        output = model(noise)

    return output


def init_weights_normal(model, mean, std):
    for p in model.parameters():
        nn.init.normal_(p, mean, std)