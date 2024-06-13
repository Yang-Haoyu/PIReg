from torch import nn
import torch
from methods.helper_fnc import create_decoder
from methods.ssl_base import ssl_base


class vime(ssl_base):
    def __init__(self, d_in,  hpara, mode = None):
        super().__init__(d_in,  hpara, mode = mode)

        self.corrup_prob = hpara["vime_corruption"]
        alpha = hpara["vime_alpha"]


        self.alpha = alpha


        self.h = None

        # ====================== ssl part ==============================

        self.mask_estimator = create_decoder(self.d_h, self.d_hid, d_in, num_layers = self.num_layers_proj)
        self.proj = create_decoder(self.d_h, self.d_hid, self.d_in, num_layers=self.num_layers_proj)


        self.reconstr_loss = nn.MSELoss()
        self.mask_loss = nn.BCEWithLogitsLoss()
        self.use_proj = True


    def mask_generator (self, corrup_prob, x):
        """Generate mask vector.
        
        Args:
            - p_m: corruption probability
            - x: feature matrix
            
        Returns:
            - mask: binary mask matrix 
        """
        # mask = self.rng.binomial(1, corrup_prob, x.shape)
        # torch.ones(x.shape)*corrup_prob
        
        return torch.bernoulli(torch.ones(x.shape, device = self.device)*corrup_prob, generator = self.tng)
     
    def pretext_generator (self, x, mask):  
        """Generate corrupted samples.
        
        Args:
            m: mask matrix
            x: feature matrix
            
        Returns:
            m_new: final mask matrix after corruption
            x_tilde: corrupted feature matrix
        """

        # Parameters

        # Randomly (and column-wise) shuffle data
        # x_bar = torch.zeros([no, dim], device=self.device)
        # for i in range(dim):
        #     idx = self.rng.permutation(no)
        #     x_bar[:, i] = x[idx, i]
        # this implementation is more efficient
        indices = torch.argsort(torch.rand_like(x), dim=0)
        x_bar = torch.gather(x, dim=0, index=indices)
        
        # Corrupt samples
        x_tilde = x * (1-mask) + x_bar * mask  
        # Define new mask matrix
        mask_new = 1 * (x != x_tilde)

        return mask_new, x_tilde
    
    def _loss(self, x):
        # generating mask: mask_label and corrupted input: x_tilde

        h, p = self.forward(x)

        # h_reg, h_corrupted = h
        mask_label, mask_est, x_est = p

        loss = self.reconstr_loss(x_est, x) + self.alpha * self.mask_loss(mask_est, mask_label.float())

        return loss



    def return_representation(self, x):
        mask = self.mask_generator(self.corrup_prob, x)
        mask_label, x_tilde = self.pretext_generator(x, mask)

        h = self.encoder(x_tilde)
        return (mask_label, h)

    def return_projs(self, h_lst):
        mask_label, h = h_lst
        mask_est = self.mask_estimator(h)
        x_est = self.proj(h)

        return (mask_label, mask_est, x_est)

