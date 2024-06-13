from torch import nn
import torch
from methods.helper_fnc import  create_encoder, create_decoder
from methods.model_base import base

class clf_mlp(base):
    def __init__(self, d_in, hpara, encoder = None, mode = "GenD", stop_gradient = False):


        d_hid, d_h = hpara["d_hid"], hpara["d_h"]
        d_out = hpara["d_out"]
    
        super().__init__(hpara)

        self.stop_gradient = stop_gradient

        # ====================== ssl part ==============================
        num_layers_enc = hpara["num_layers_enc"]
        num_layers_clf = hpara["num_layers_clf"]

        if encoder is None:
            self.encoder = create_encoder(d_in, d_hid, d_h, num_layers = num_layers_enc)
        else:
            self.encoder = encoder
        self.predictor = create_decoder(d_h, d_hid, d_out, num_layers = num_layers_clf)

        # ====================== optimization ==============================
        self.loss = nn.CrossEntropyLoss() 
        self.mode = mode

    def forward(self, x):

        h =  self.encoder(x)
        if self.stop_gradient:
            h=h.detach()
        out = self.predictor(h)
        return out
    
    def compute_loss(self, _input):
        x_reg, x_priv, y = _input


        if self.mode == "GenD":
            x = x_priv
        elif self.mode == "Reg":
            x = x_reg
        elif self.mode == "PFD":
            x = torch.cat([x_priv, x_reg], dim = -1)
        elif self.mode == "Priv":
            x = x_priv
        else:
            raise NotImplementedError
        #  Concatenate original data with itself to be used when computing reconstruction error 
        y_hat = self.forward(x)
        if len(y) == 2:
            loss =  self.loss(y_hat, y[0])
        else:
            loss =  self.loss(y_hat, y)

        return loss
    
    def return_test_input(self, batch):


        x_reg, x_priv, y = batch

        if self.mode == "GenD":
            x = x_priv
        elif self.mode == "Reg":
            x = x_reg
        elif self.mode == "PFD":
            x = torch.cat([x_priv, x_reg], dim = -1)
        elif self.mode == "Priv":
            x = x_priv
        else:
            raise NotImplementedError

        return (x, y)
    
    @torch.inference_mode()
    def get_embeddings(self, x):
        return self.encoder(x)

    @torch.inference_mode()
    def predict_logits(self, x):
        h =  self.encoder(x)
        out = self.predictor(h)
       
        return  out
    @torch.inference_mode()
    def predict_label(self, x):
        h =  self.encoder(x)
        logits = self.predictor(h)
        y_pred_prob = torch.softmax(logits, dim = -1).cpu()
        y_pred = torch.argmax(y_pred_prob, dim = -1)
        return y_pred
