from torch import nn
import torch
import numpy as np
from methods.helper_fnc import  create_decoder
import math
import torch.nn.functional as F
from methods.losses import off_diagonal
from methods.ssl_base import ssl_base


def bt_loss_bs(p, z, lambd=0.01, normalize=False):
    #barlow twins loss but in batch dims
    c = torch.matmul(F.normalize(p), F.normalize(z).T)
    assert c.min()>-1 and c.max()<1
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    if normalize: loss = loss/p.shape[0]
    return loss

def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items

class lfr(ssl_base):
    def __init__(self, d_in,  hpara,   mode = None):
        super().__init__(d_in,  hpara,   mode = mode)


        self.d_in = d_in

        self.num_targets = hpara["lfr_num_targets"]
        target_sample_ratio = hpara["lfr_target_sample_ratio"]
        sample_data = hpara["sample_data"]


        self.hpara = hpara

        target_encoders=[]
        predictors=[]
        
        for i in range(self.num_targets):
            with torch.no_grad():
                # randomly initialize target networks with no gradient 
                for _ in range(target_sample_ratio):
                    target = create_decoder(d_in, self.d_hid, self.d_proj, num_layers = self.num_layers_enc+self.num_layers_proj)
                    for param_k in target.parameters():
                        param_k.requires_grad = False  # not update by gradient
                    target_encoders.append(target)
            predictor = create_decoder(self.d_h, self.d_hid, self.d_proj, num_layers = self.num_layers_proj)
            predictors.append(predictor)
        
        if target_sample_ratio > 1:
            print("===============selecting {} target encoders from {}===============".format(self.num_targets, len(target_encoders)))
            target_encoders = self.select_targets(target_encoders, self.num_targets, sample_data)

        self.predictors = nn.ModuleList(predictors)
        self.target_encoders = nn.ModuleList(target_encoders)



        self.bn = nn.BatchNorm1d(self.d_hid, affine=False)
        
        # ====================== semi part ==============================

        self.criterion = nn.CosineSimilarity(dim=1)
    def select_targets(self, target_encoders, num_targets, sample_data):
        '''
        select num_targets number of encoders out of target_encoders
        ''' 
        with torch.no_grad():
            sims = []
            for t in target_encoders:
                # (bs, dim)
                rep = t(sample_data)
                if rep.shape[0] > 1000: 
                    rep = rep[np.random.RandomState(seed=42).permutation(np.arange(rep.shape[0]))[:1000]]
                rep_normalized = F.normalize(rep, dim=1)
                # (bs, bs) cosine similarity
                sim = rep_normalized @ rep_normalized.T
                sims.append(sim.view(-1))
            # N, bs^2
            sims = torch.stack(sims)
            sims_normalized = F.normalize(sims, dim=1)
            # N,N
            sims_targets = sims_normalized @ sims_normalized.T
            result = dpp(sims_targets.cpu().numpy(), num_targets)
        return [target_encoders[idx] for idx in result]



    def forward(self, x):
        z_w = self.encoder(x) # NxC

        target_reps = []
        predicted_reps = []
        for i in range(self.num_targets):
            target = self.target_encoders[i]
            predictor = self.predictors[i]
            z_a = target(x) # NxC
            p_a = predictor(z_w)
            target_reps.append(z_a)
            predicted_reps.append(p_a)

        return (predicted_reps, target_reps)
    
    def _loss(self, x):
        #  Concatenate original data with itself to be used when computing reconstruction error 
        predicted_reps, target_reps =self.forward(x)

        loss = torch.tensor(0).to(self.device)
        for t in range(self.num_targets):
            p = predicted_reps[t]
            z = target_reps[t]
            
            loss = loss + bt_loss_bs(p,z, lambd=self.hpara["lfr_lambd"])

        loss = loss/self.num_targets 

        return loss
    