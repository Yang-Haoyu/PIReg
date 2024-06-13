from torch import nn
import torch
import numpy as np
from methods.losses import NTXent
from torch.distributions.uniform import Uniform
from methods.helper_fnc import  create_encoder, create_decoder
from methods.losses import  off_diagonal
from methods.siams import CovarianceLoss
import torch.nn.functional as F

class cross_loss(nn.Module):
    def __init__(self):
        super(cross_loss, self).__init__()

    def forward(self, x, z, normed = False):

        # empirical cross-correlation matrix
        N, D = z.size()

        mu_x = torch.mean(x, 0)
        mu_z = torch.mean(z, 0)
        z_hat =  z - mu_z
        x_hat =  x - mu_x

        c = (z_hat.T @ x_hat)/N 

        # sum the cross-correlation matrix between all gpus
        if normed:
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_z = torch.sqrt(z.var(dim=0) + 0.0001)
            denom = std_z.unsqueeze(1) @std_x.unsqueeze(0)
            c.div_(len(z)).div_(denom)
        else:
            c.div_(len(z))


        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + off_diag

        return loss



class pcorinfomax(nn.Module):
    def __init__(self,  hpara, teacher):
        super().__init__()

        seed =  hpara["seed"]
        num_layers_enc = hpara["num_layers_enc"]
        num_layers_proj = hpara["num_layers_proj"]
        corruption_rate = hpara["corruption_scarf"]
        d_in, d_in_priv = hpara["d_reg"], hpara["d_priv"]
        d_hid, d_h, d_proj = hpara["d_hid"], hpara["d_h"], hpara["d_proj"]
        self.loss_comp = hpara["loss_comp"]

        self.rng = np.random.default_rng(seed)
        if torch.cuda.is_available():
            self.tng = torch.Generator('cuda').manual_seed(seed)
        else:
            self.tng = torch.Generator().manual_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.corruption_rate = corruption_rate

        self.covariance_loss = CovarianceLoss(hpara=hpara)
        # self.cross_cov_losss = CrossCovarianceLoss(hpara=hpara)
        
        self.marginals = Uniform(torch.Tensor(hpara["Reg_min"]), torch.Tensor(hpara["Reg_max"]))


        self.d_in = d_in

        # ====================== ssl part ==============================
        self.encoder = create_encoder(self.d_in, d_hid, d_h, num_layers = num_layers_enc)
        self.priv_encoder = create_encoder(d_in_priv, d_hid, d_h, num_layers = num_layers_enc)

        self.proj= create_decoder(d_h, d_hid, d_proj, num_layers = num_layers_proj)
        self.proj_priv = create_decoder(d_h, d_hid, d_proj, num_layers = num_layers_proj)


        self.teacher = teacher
        self.cross_loss = cross_loss()

        # self.ntxent_loss = NTXent()
        self.pdist = nn.PairwiseDistance(p=2)
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

    
    def compute_loss(self, x_reg, x_priv):
        #  Concatenate original data with itself to be used when computing reconstruction error 
        h_reg, h_corrupted = self.learn_representation(x_reg)
        # _, h_corrupted = self.learn_representation(x_reg)

        # h_priv = self.priv_encoder(x_priv)

        p_reg = self.proj(h_reg)
        p_corrupted = self.proj(h_corrupted)


        h_priv_teacher = self.teacher.encoder(x_priv).detach()

        if "proj" in self.loss_comp:
            p_priv_teacher = self.proj_priv(h_priv_teacher)
        else:
            p_priv_teacher = h_priv_teacher


        repr_loss = F.mse_loss(p_reg, p_corrupted)
        
        cov_loss = self.covariance_loss(p_reg, p_corrupted)

        loss =   repr_loss + cov_loss

        if "closs" in self.loss_comp:
            closs = self.cross_loss(p_priv_teacher, p_corrupted) 
            loss = loss + closs
        if "closs cor" in self.loss_comp:
            closs = self.cross_loss(p_priv_teacher, p_corrupted, normed = True) 
            loss = loss + closs
        if "dloss" in self.loss_comp:
            dloss = self.pdist(p_reg, h_priv_teacher).mean() + self.pdist(p_corrupted, h_priv_teacher).mean()
            loss = loss + dloss
        if "priv_cov_loss" in self.loss_comp:
            priv_cov_loss = self.covariance_loss(p_priv_teacher, p_priv_teacher)
            loss = loss + priv_cov_loss



        # loss =   repr_loss + cov_loss  
        return loss

    

    def get_embeddings(self, x):
        return self.encoder(x)

class pvime(nn.Module):
    def __init__(self, hpara, teacher):
        super().__init__()
        seed =  hpara["seed"]
        num_layers_enc = hpara["num_layers_enc"]
        num_layers_proj = hpara["num_layers_proj"]
        corrup_prob = hpara["vime_corruption"]
        class_weights = hpara["class_weights"]
        alpha = hpara["vime_alpha"]
        d_hid, d_h, d_proj = hpara["d_hid"], hpara["d_h"], hpara["d_proj"]
        d_in, d_in_priv = hpara["d_reg"], hpara["d_priv"]
        self.loss_comp = hpara["loss_comp"]

        self.alpha = alpha
        self.corrup_prob = corrup_prob
        # ============================== control randomness ========================
        self.rng = np.random.default_rng(seed)
        if torch.cuda.is_available():
            self.tng = torch.Generator('cuda').manual_seed(seed)
        else:
            self.tng = torch.Generator().manual_seed(seed)

        self.teacher = teacher
        self.pdist = nn.PairwiseDistance(p=2)

        self.h = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ====================== ssl part ==============================
        self.encoder = create_encoder(d_in, d_hid, d_h, num_layers = num_layers_enc)
        self.mask_estimator = create_decoder(d_h, d_hid, d_in, num_layers = num_layers_proj)
        self.proj_reconstr = create_decoder(d_h, d_hid, d_in, num_layers=num_layers_proj)

        self.proj_priv = create_decoder(d_h, d_hid, d_proj, num_layers = num_layers_proj)
        self.priv_encoder = create_encoder(d_in_priv, d_hid, d_h, num_layers = num_layers_enc)


        self.reconstr_loss = nn.MSELoss()
        self.mask_loss = nn.BCEWithLogitsLoss()
        self.covariance_loss = CovarianceLoss(hpara=hpara)

        self.cross_loss = cross_loss()


        if class_weights is None:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

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
    
    def compute_loss(self, x, x_priv):
        # generating mask: mask_label and corrupted input: x_tilde
        mask = self.mask_generator(self.corrup_prob, x)
        mask_label, x_tilde = self.pretext_generator(x, mask)

        self.h = self.encoder(x_tilde)
        mask_est = self.mask_estimator(self.h)
        x_est = self.proj_reconstr(self.h)


        h_priv_teacher = self.teacher.encoder(x_priv).detach()

        if "proj" in self.loss_comp:
            p_priv_teacher = self.proj_priv(h_priv_teacher)
        else:
            p_priv_teacher = h_priv_teacher

        loss = self.reconstr_loss(x_est, x) + self.alpha * self.mask_loss(mask_est, mask_label.float())

        if "closs" in self.loss_comp:
            closs = self.cross_loss(p_priv_teacher, self.h) 
            loss = loss + closs
        if "closs cor" in self.loss_comp:
            closs = self.cross_loss(p_priv_teacher, self.h, normed = True) 
            loss = loss + closs
        if "dloss" in self.loss_comp:
            dloss = self.pdist(self.h, h_priv_teacher).mean()
            loss = loss + dloss
        if "cov_loss" in self.loss_comp:
            cov_loss = self.covariance_loss(p_priv_teacher, p_priv_teacher)
            loss = loss + cov_loss

        return loss



    def learn_representation(self, x):
        # mask = self.mask_generator(self.corrup_prob, x)
        # mask_label, x_tilde = self.pretext_generator(x, mask)

        # h = self.encoder(x_tilde)

        return (self.h,)


