from torch import nn
import torch
import numpy as np
from torch.distributions.uniform import Uniform
from methods.helper_fnc import  create_encoder, create_decoder
from methods.losses import  off_diagonal
import torch.nn.functional as F
from methods.siams import CovarianceLoss

class TriDeNT(nn.Module):
    def __init__(self,  hpara, teacher):
        super().__init__()

        seed =  hpara["seed"]
        num_layers_enc = hpara["num_layers_enc"]
        num_layers_proj = hpara["num_layers_proj"]
        corruption_rate = hpara["corruption_scarf"]
        d_in, d_in_priv = hpara["d_reg"], hpara["d_priv"]
        d_hid, d_h, d_proj = hpara["d_hid"], hpara["d_h"], hpara["d_proj"]

        self.rng = np.random.default_rng(seed)
        if torch.cuda.is_available():
            self.tng = torch.Generator('cuda').manual_seed(seed)
        else:
            self.tng = torch.Generator().manual_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.corruption_rate = corruption_rate

        self.marginals = Uniform(torch.Tensor(hpara["Reg_min"]), torch.Tensor(hpara["Reg_max"]))

        self.d_in = d_in

        # ====================== ssl part ==============================
        self.encoder = create_encoder(self.d_in, d_hid, d_h, num_layers = num_layers_enc)
        self.priv_encoder = create_encoder(d_in_priv, d_hid, d_h, num_layers = num_layers_enc)
        self.proj= create_decoder(d_h, d_hid, d_proj, num_layers = num_layers_proj)
        self.proj_priv = create_decoder(d_h, d_hid, d_proj, num_layers = num_layers_proj)

        self.teacher = teacher

        self.covariance_loss = CovarianceLoss(hpara=hpara)
        self.bn = nn.BatchNorm1d(d_proj, affine=False)
    def learn_representation(self, x):
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
    

    def forward(self, x):
        h_reg, h_corrupted = self.learn_representation(x)

        p_reg = self.proj(h_reg)
        p_corrupted = self.proj(h_corrupted)
        return (h_reg, h_corrupted, p_reg, p_corrupted)
    def cov_loss(self, x):
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (x.shape[0] - 1)

        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            x.shape[1]
        ) 
        return cov_loss
    def std_loss(self, x):
        x = x - x.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)

        std_loss = torch.mean(F.relu(1 - std_x)) / 2 
        return std_loss


    def cross_loss(self, x, z):
        #  Concatenate original data with itself to be used when computing reconstruction error 

        # empirical cross-correlation matrix
        N, D = z.size()

        mu_x = torch.mean(x, 0)
        mu_z = torch.mean(z, 0)
        z_hat =  z - mu_z
        x_hat =  x - mu_x

        c = (z_hat.T @ x_hat)/N 

        # sum the cross-correlation matrix between all gpus
        c.div_(len(z))


        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + off_diag

        return loss
    def compute_loss(self, x, x_priv):
        #  Concatenate original data with itself to be used when computing reconstruction error 
        h_reg, h_corrupted, p_reg, p_corrupted = self.forward(x)
        repr_loss = F.mse_loss(p_reg, p_corrupted)

        h_priv = self.priv_encoder(x_priv)
        p_priv = self.proj_priv(h_priv)

        # loss = repr_loss +  std_loss +  cov_loss  + self.covariance_loss(self.proj2(h_reg), h_priv)  + self.covariance_loss(self.proj2(h_corrupted), h_priv) 
        loss = repr_loss +  self.std_loss(p_reg) +  self.std_loss(p_corrupted) +  self.cov_loss(p_reg) + self.cov_loss(p_corrupted) 
        loss = loss +  F.mse_loss(p_priv, p_corrupted) +  F.mse_loss(p_reg, p_priv)
        loss = loss +  self.std_loss(p_priv) +  self.cov_loss(p_priv)

        return loss


    

    def get_embeddings(self, x):
        return self.encoder(x)

