import torch
from methods.model_base import base


class priv_ssl(base):
    def __init__(self, ssl_backbone, hpara):


        super().__init__( hpara)

        "==============paras=================="
        self.lr = hpara["lr"]

        "==============SSL base models=================="
        self.ssl_backbone = ssl_backbone



    def compute_loss(self, _input):
        x_reg, x_priv, _ = _input

        # get the projections from regular inputs
        loss = self.ssl_backbone.compute_loss(x_reg, x_priv)

        return loss

    # @torch.inference_mode()
    def encoder(self, x):
        return self.ssl_backbone.encoder(x)
