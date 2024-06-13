from torch import nn
import matplotlib.pyplot as plt
import lightning as L
import torch
import numpy as np
from methods.metrics import compute_scores_cifar, compute_scores_nsqip, compute_scores_mnist, compute_scores_mover

from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from sklearn.linear_model import LogisticRegression
from  torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryAveragePrecision


class base(L.LightningModule):
    
    def __init__(self, hpara):
        super().__init__()
        self.seed = hpara["seed"]
        self.dataset = hpara["dataset"]
        self.lr = hpara["lr"]

        self.rng = np.random.default_rng(self.seed)
        #         self.tng = torch.Generator(seed)
        if torch.cuda.is_available():
            self.tng = torch.Generator('cuda').manual_seed(self.seed)
        else:
            self.tng = torch.Generator().manual_seed(self.seed)
        self.loss_lst_val = []
        self.loss_lst_tr = []
        


    def forward(self):
        # return embeddings
        print("Please implement forward method")
        raise NotImplementedError
    
        
    def compute_loss(self, batch):
        # combut loss, given the batched input
        print("Please implement compute_loss method")
        raise NotImplementedError
    
    def predict_logits(self, x):
        print("Please implement predict_logits method")
        raise NotImplementedError
    
    def return_test_input(self, batch):
        x_reg, x_priv, y = batch
        return (x_reg, y)

    def training_step(self, batch, batch_idx):
        # compute loss for the labeled data
        loss = self.compute_loss(batch)
        self.loss_lst_tr.append(loss.item())

        # logging
        self.log("train_batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            
        return loss

    def on_train_epoch_end(self):
        self.log("train_loss", np.mean(self.loss_lst_tr),  on_epoch=True, prog_bar=True, logger=True)
        self.loss_lst_tr.clear()

    def validation_step(self, batch, batch_idx):
        # compute validation loss for the labeled data 
        loss = self.compute_loss(batch)

        # logging 
        self.loss_lst_val.append(loss.item())
        self.log("val_batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        self.log("val_loss", np.mean(self.loss_lst_val),  on_epoch=True, prog_bar=True, logger=True)
        self.loss_lst_val.clear()

    def test_step(self, batch, batch_idx):
        # compute test loss
        x_tst, y_tst = self.return_test_input(batch)

        # compute metrics
        logits = self.predict_logits(x_tst).detach()
        y_pred_prob = torch.softmax(logits, dim = -1).cpu()
        y_tst = y_tst.cpu()

        if self.dataset == "cifar":
            scores = compute_scores_cifar(y_pred_prob, y_tst)
        elif self.dataset == "nsqip":
            scores = compute_scores_nsqip(y_pred_prob, y_tst)
        elif self.dataset == "mnist":
            scores = compute_scores_mnist(y_pred_prob, y_tst)
        elif self.dataset == "mover":
            scores = compute_scores_mover(y_pred_prob, y_tst)
        else:
            raise NotImplementedError

        # logging
        for k,v in scores.items():
            self.log("test_{}".format(k), v,  on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min'
            )
        

    #     return optimizer, scheduler
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "frequency": 1,
        },
    }

