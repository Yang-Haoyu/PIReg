from torch import nn
import torch
import numpy as np
from methods.helper_fnc import  create_encoder, create_decoder
from methods.model_base import base

class tram(base):
    def __init__(self, hpara):
        super().__init__(hpara)
        seed =  hpara["seed"]
        num_layers_enc = hpara["num_layers_enc"]
        class_weights = hpara["class_weights"]
        num_layers_clf = hpara["num_layers_clf"]

        d_hid, d_h, d_proj = hpara["d_hid"], hpara["d_h"], hpara["d_proj"]
        d_in, d_in_priv = hpara["d_reg"], hpara["d_priv"]
        d_out = hpara["d_out"]

        # ============================== control randomness ========================
        self.rng = np.random.default_rng(seed)
        if torch.cuda.is_available():
            self.tng = torch.Generator('cuda').manual_seed(seed)
        else:
            self.tng = torch.Generator().manual_seed(seed)

        # ====================== ssl part ==============================
        self.encoder = create_encoder(d_in, d_hid, d_h, num_layers = num_layers_enc)
        self.hybrid_encoder = create_encoder(d_in_priv + d_h, d_hid, d_h, num_layers = num_layers_enc)

        self.hybrid_pred_head = create_decoder(d_h, d_hid, d_out, num_layers=num_layers_clf)
        self.pred_head = create_decoder(d_h, d_hid, d_out, num_layers=num_layers_clf)



        if class_weights is None:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    def compute_loss(self, _input):
        x, x_priv, y = _input

        h = self.encoder(x)
        phi = self.hybrid_encoder(torch.cat([h, x_priv], dim = -1))
        
        logits_priv = self.hybrid_pred_head(phi)
        logits_x = self.pred_head (h.detach())

        if len(y) == 2:
            loss =  self.criterion(logits_priv, y[0]) + self.criterion(logits_x, y[0])
        else:
            loss =  self.criterion(logits_priv, y) + self.criterion(logits_x, y)



        return loss


    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)

    @torch.inference_mode()
    def predict_logits(self, x):
        h =  self.encoder(x)
        out = self.pred_head(h)
       
        return  out
    @torch.inference_mode()
    def predict_label(self, x):
        h =  self.encoder(x)
        logits = self.pred_head(h)
        y_pred_prob = torch.softmax(logits, dim = -1).cpu()
        y_pred = torch.argmax(y_pred_prob, dim = -1)
        return y_pred



