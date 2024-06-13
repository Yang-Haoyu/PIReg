from torch import nn
import torch
import numpy as np
from torch.distributions.uniform import Uniform
from methods.helper_fnc import  create_encoder, create_decoder
from methods.losses import  off_diagonal
import torch.nn.functional as F
from methods.ssl_base import corrupted_ssl
from methods.losses import  NTXent

class SCARF(corrupted_ssl):
    def __init__(self, d_in,  hpara, features_low, features_high, mode = None):
        super().__init__(d_in,  hpara, features_low, features_high, mode)


        self.ntxent_loss = NTXent()

    
    def _loss(self, x):
        #  Concatenate original data with itself to be used when computing reconstruction error 
        h, p = self.forward(x)

        # h_reg, h_corrupted = h
        p_reg, p_corrupted = p

        loss = self.ntxent_loss(p_reg, p_corrupted)

        return loss



class vicreg(corrupted_ssl):
    def __init__(self, d_in,  hpara, features_low, features_high, mode = None):
        super().__init__(d_in,  hpara, features_low, features_high, mode)


    def _loss(self, x):
        h, p = self.forward(x)
        # h_reg, h_corrupted = h
        p_reg, p_corrupted = p
    
        #  Concatenate original data with itself to be used when computing reconstruction error 
        repr_loss = F.mse_loss(p_reg, p_corrupted)

        x = p_reg
        y = p_corrupted

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (y.T @ y) / (x.shape[0] - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            x.shape[1]
        ) + off_diagonal(cov_y).pow_(2).sum().div(x.shape[1])

        loss = repr_loss +  std_loss +  cov_loss

        return loss
    


class simsiam(corrupted_ssl):
    def __init__(self, d_in,  hpara, features_low, features_high, mode = None):
        super().__init__(d_in,  hpara, features_low, features_high, mode)

        self.criterion = nn.CosineSimilarity(dim=1)

    
    def _loss(self, x):

        h, p = self.forward(x)

        h_reg, h_corrupted = h
        p_reg, p_corrupted = p

        p1, p2, z1, z2 = p_reg, p_corrupted,  h_reg.detach(), h_corrupted.detach()
    

        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5

        return loss
    


class barlowtwins(corrupted_ssl):
    def __init__(self, d_in,  hpara, features_low, features_high, mode = None):
        super().__init__(d_in,  hpara, features_low, features_high, mode)


        self.bn = nn.BatchNorm1d(self.d_proj, affine=False)
        

        self.criterion = nn.CosineSimilarity(dim=1)

    
    def _loss(self, x):
        #  Concatenate original data with itself to be used when computing reconstruction error 
        h, p = self.forward(x)

        # h_reg, h_corrupted = h
        p1, p2 = p

        # empirical cross-correlation matrix
        c = self.bn(p1).T @ self.bn(p2)

        # sum the cross-correlation matrix between all gpus
        c.div_(len(p1))


        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag +  off_diag

        return loss
    

class CovarianceLoss(nn.Module):
    """Big-bang factor of CorInfoMax Loss: loss calculation from outputs of the projection network,
    z1 (NXD) from the first branch and z2 (NXD) from the second branch. Returns loss part comes from bing-bang factor.
    """
    def __init__(self, hpara):
        super(CovarianceLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.R1 = torch.eye(hpara["d_proj"] , dtype=torch.float64,  device=self.device, requires_grad=False)
        self.mu1 = torch.zeros(hpara["d_proj"], dtype=torch.float64, device=self.device, requires_grad=False)
        self.R2 = torch.eye(hpara["d_proj"] , dtype=torch.float64, device=self.device,  requires_grad=False)
        self.mu2 = torch.zeros(hpara["d_proj"], dtype=torch.float64, device=self.device, requires_grad=False)
        self.new_R1 = torch.zeros((hpara["d_proj"], hpara["d_proj"]), dtype=torch.float64,  device=self.device, requires_grad=False) 
        self.new_mu1 = torch.zeros(hpara["d_proj"], dtype=torch.float64, device=self.device, requires_grad=False) 
        self.new_R2 = torch.zeros((hpara["d_proj"], hpara["d_proj"]), dtype=torch.float64, requires_grad=False) 
        self.new_mu2 = torch.zeros(hpara["d_proj"], dtype=torch.float64, device=self.device, requires_grad=False) 
        self.la_R = hpara["infomax_la_R"]
        self.la_mu = hpara["infomax_la_mu"]
        self.R_eigs = torch.linalg.eigvals(self.R1).unsqueeze(0)
        self.R_eps_weight = hpara["infomax_R_eps_weight"]
        self.R_eps = self.R_eps_weight*torch.eye(hpara["d_proj"], dtype=torch.float64, device=self.device, requires_grad=False)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        la_R = self.la_R
        la_mu = self.la_mu

        N, D = z1.size()

        # mean estimation
        mu_update1 = torch.mean(z1, 0)
        mu_update2 = torch.mean(z2, 0)
        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu_update1)
        self.new_mu2 = la_mu*(self.mu2) + (1-la_mu)*(mu_update2)

        # covariance matrix estimation
        z1_hat =  z1 - self.new_mu1
        z2_hat =  z2 - self.new_mu2
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N
        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

        # loss calculation 
        cov_loss = - (torch.logdet(self.new_R1 + self.R_eps) + torch.logdet(self.new_R2 + self.R_eps)) / D

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.mu1 = self.new_mu1.detach()
        self.R2 = self.new_R2.detach()
        self.mu2 = self.new_mu2.detach()

        return cov_loss

    def save_eigs(self) -> np.array: 
        with torch.no_grad():
            R_eig = torch.linalg.eigvals(self.R1).unsqueeze(0)
            self.R_eigs = torch.cat((self.R_eigs, R_eig), 0)
            R_eig_arr = np.real(self.R_eigs).cpu().detach().numpy()
        return R_eig_arr 


class corinfomax(corrupted_ssl):
    def __init__(self, d_in,  hpara, features_low, features_high, mode = None):
        super().__init__(d_in,  hpara, features_low, features_high, mode)


        self.covariance_loss = CovarianceLoss(hpara=hpara)

    
    def _loss(self, x):
        #  Concatenate original data with itself to be used when computing reconstruction error 
        h, p = self.forward(x)

        # h_reg, h_corrupted = h
        p_reg, p_corrupted = p
        repr_loss = F.mse_loss(p_reg, p_corrupted)

        cov_loss = self.covariance_loss(p_reg, p_corrupted)
        loss = repr_loss +  cov_loss

        return loss
