import torch
from methods.helper_fnc import  create_encoder, create_decoder
from methods.model_base import base
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        loss = nn.KLDivLoss(reduction="batchmean")(p_s,p_t)* (self.T**2)
        return loss

class LUPI_KD(base):

    def __init__(self, d_in, hpara, gamma = 0.5, T=1.0, encoder = None, stop_gradient = False):
        
        super().__init__(hpara)

        num_layers_enc = hpara["num_layers_enc"]
        num_layers_clf = hpara["num_layers_clf"]
        d_hid, d_h = hpara["d_hid"], hpara["d_h"]
        d_out = hpara["d_out"]

        "----------------------- model structure ---------------------"
        if encoder is None:
            # If there is no pretrained encoder, create a new one to match the complexity
            self.encoder = create_encoder(d_in, d_hid,  d_h, num_layers = num_layers_enc)
        else:
            self.encoder = encoder
        self.predictor =create_decoder(d_h, d_hid, d_out, num_layers = num_layers_clf)

        "----------------------- Hyperparameters ---------------------"
        self.stop_gradient = stop_gradient
        self.gamma = gamma
        self.T = T


        "----------------------- Losses ---------------------"


        self.div = DistillKL(self.T)

        if hpara["class_weights"] is None:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            # class_weights=torch.tensor([hpara["class_weights"][0], hpara["class_weights"][1]],dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=hpara["class_weights"], reduction='mean')

    
    def forward(self, x):

        h = self.encoder(x)

        if self.stop_gradient:
            # to fine tune SSL model
            h = h.detach()
        # ==================== get the supervised loss ==============
        logits = self.predictor(h)
        return logits
    
    def compute_loss(self, _input):
        x_reg, _, y_lst = _input
        y_gt, logits_teacher = y_lst
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """ ================== Step: Predict Logits for labeled data ==================== """
        logits_student = self.forward(x_reg)


        """ ================== Step: Get Teacher's Knowledge for labeled data ==================== """

        loss_ce = self.criterion(logits_student, y_gt)

        loss_dist = self.div(logits_student, logits_teacher)


        loss = loss_ce + self.gamma*loss_dist

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.lr)
        return optimizer
    
    @torch.inference_mode()
    def predict_logits(self, x):

        h = self.encoder(x)

        # ==================== get the supervised loss ==============
        logits = self.predictor(h)

        return logits
    @torch.inference_mode()
    def predict_label(self, x):
        h = self.encoder(x)

        # ==================== get the supervised loss ==============
        logits = self.predictor(h)

        y_pred_prob = torch.softmax(logits, dim = -1).cpu()
        y_pred = torch.argmax(y_pred_prob, dim = -1)
        return y_pred