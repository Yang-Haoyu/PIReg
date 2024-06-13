from torch import nn
import torch
import numpy as np
from torch.distributions.uniform import Uniform
from methods.helper_fnc import  create_encoder, create_decoder
from methods.losses import  off_diagonal
import torch.nn.functional as F
from methods.model_base import base

class ssl_base(base):
    def __init__(self, d_in,  hpara, features_low = None, features_high = None, mode = None):
        super().__init__( hpara)

        seed =  hpara["seed"]
        self.num_layers_enc = hpara["num_layers_enc"]
        self.num_layers_proj = hpara["num_layers_proj"]
        

        self.d_in = d_in
        self.d_hid, self.d_h, self.d_proj = hpara["d_hid"], hpara["d_h"], hpara["d_proj"]

        self.use_proj = hpara["use_proj"]

        self.rng = np.random.default_rng(seed)
        if torch.cuda.is_available():
            self.tng = torch.Generator('cuda').manual_seed(seed)
        else:
            self.tng = torch.Generator().manual_seed(seed)



        if features_low is not None:
            self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))

        
        self.mode = mode
        # ====================== ssl part ==============================
        self.encoder = create_encoder(self.d_in, self.d_hid, self.d_h, num_layers = self.num_layers_enc)
        self.proj= create_decoder(self.d_h, self.d_hid, self.d_proj, num_layers = self.num_layers_proj)


    def return_representation(self, x):
        raise NotImplementedError
    def return_projs(self, h):
        raise NotImplementedError
    
    def forward(self, x):
        h = self.return_representation(x)

        if self.use_proj:
            p = self.return_projs(h)
            return (h, p)
        else:
            return (h, h)

    
    def _loss(self, x):
        raise NotADirectoryError
    

    def compute_loss(self, _input):
        x_reg, x_priv, _ = _input

        if self.mode == "Priv":
            x = x_priv
        elif self.mode == "Reg":
            x = x_reg
        elif self.mode == "PFD":
            x = torch.cat([x_priv, x_reg], dim = -1)
        else:
            raise NotImplementedError
        
        # get the projections from regular inputs
        loss = self._loss(x)

        return loss

    def get_embeddings(self, x):
        return self.encoder(x)


class corrupted_ssl(ssl_base):
    def __init__(self, d_in,  hpara, features_low = None, features_high = None, mode = None):
        super().__init__(d_in,  hpara, features_low , features_high, mode)

    
        corruption_rate = hpara["corruption_scarf"]
        self.corruption_rate = corruption_rate


        
    def return_representation(self, x):
        batch_size, m = x.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the jth column to True at random, such that corruption_len / m = corruption_rate
        # 2: create a random tensor of size (batch size, m) drawn from the uniform distribution defined by the min, max values of the training set
        # 3: replace x_corrupted_ij by x_random_ij where mask_ij is true

        corruption_mask = torch.bernoulli(torch.ones(x.shape, device = self.device)*self.corruption_rate, generator = self.tng)
        # marginal noise
        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        # corrupted input
        x_corrupted = x_random*corruption_mask + (1-corruption_mask)*x


        # get embeddings
        h_reg = self.encoder(x)
        h_corrupted = self.encoder(x_corrupted)

        return (h_reg, h_corrupted)
    
    def return_projs(self, h):
        h_reg, h_corrupted = h
        p_reg = self.proj(h_reg)
        p_corrupted = self.proj(h_corrupted)
        return (p_reg, p_corrupted)