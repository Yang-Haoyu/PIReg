
from torch import nn
import torch

class MLP(torch.nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim, num_layers: int, dropout: float = 0.0) -> None:
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            # layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_dim))

        super().__init__(*layers)

def create_decoder(d_in, d_hid, d_outs, num_layers = 2):
    """
    Generate the decoder and projection heads for ssl methods
    d_hid is the dim of latent representation
    d_outs is the list of the dim of the output of decoder and projection heads
    """
    if isinstance(d_outs, int):
        return MLP(d_in, d_hid, d_outs, num_layers)
    elif isinstance(d_outs, list):
        return ( MLP(d_in, d_hid, d_outs, num_layers) for i in d_outs)
    else:
        raise NotImplementedError
    

def create_encoder(d_in, d_hid, d_outs,num_layers = 3):
    """
    Generate the decoder and projection heads for ssl methods
    d_hid is the dim of latent representation
    d_outs is the list of the dim of the output of decoder and projection heads
    """
    return  MLP(d_in, d_hid, d_outs, num_layers)


