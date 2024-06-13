from torch import nn
import torch
import numpy as np
from methods.helper_fnc import  create_encoder
from methods.model_base import base




class twosteps(base):
    def __init__(self, hpara, teacher):
        super().__init__(hpara)
        seed =  hpara["seed"]
        num_layers_enc = hpara["num_layers_enc"]

        d_hid = hpara["d_hid"]
        d_in, d_in_priv = hpara["d_reg"], hpara["d_priv"]


        # ============================== control randomness ========================
        self.rng = np.random.default_rng(seed)
        if torch.cuda.is_available():
            self.tng = torch.Generator('cuda').manual_seed(seed)
        else:
            self.tng = torch.Generator().manual_seed(seed)

        # ====================== ssl part ==============================
        self.encoder = create_encoder(d_in, d_hid, d_in_priv, num_layers = 2*num_layers_enc)

        self.criterion = nn.MSELoss(reduction='mean')
        self.teacher = teacher


    def compute_loss(self, _input):
        x, x_priv, y = _input

        h = self.encoder(x)

        loss =  self.criterion(h, x_priv)

        return loss


    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)

    @torch.inference_mode()
    def predict_logits(self, x):
        h =  self.encoder(x)
        out = self.teacher(h)
       
        return  out
    @torch.inference_mode()
    def predict_label(self, x):
        h =  self.encoder(x)
        logits = self.teacher(h)
        y_pred_prob = torch.softmax(logits, dim = -1).cpu()
        y_pred = torch.argmax(y_pred_prob, dim = -1)
        return y_pred



