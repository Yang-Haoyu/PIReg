import torch
from methods.model_base import base


class ssl_model(base):
    def __init__(self, ssl_backbone, hpara, mode = None):


        super().__init__(seed = hpara["seed"], dataset = hpara["dataset"])

        "==============paras=================="
        self.lr = hpara["lr"]

        "==============SSL base models=================="
        self.ssl_backbone = ssl_backbone
        self.mode = mode


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
        loss = self.ssl_backbone.compute_loss(x)

        return loss


    # @torch.inference_mode()
    def get_embeddings(self, x):
        return self.ssl_backbone.encoder(x)
