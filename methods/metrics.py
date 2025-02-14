import torch
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from  torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryAveragePrecision



def compute_scores_cifar(y_pred_prob, y_tst):
    scores = {}
    acc = accuracy_score(torch.argmax(y_pred_prob, dim  = 1), y_tst)
    scores["acc"] = acc
    return scores

def compute_scores_mnist(y_pred_prob, y_tst):
    scores = {}
    acc = accuracy_score(torch.argmax(y_pred_prob, dim  = 1), y_tst)
    scores["acc"] = acc
    for i in range(10):
        acc = accuracy_score(torch.argmax(y_pred_prob, dim  = 1)[y_tst == i], y_tst[y_tst == i])
        scores["acc_{}".format(i)] = acc
    return scores

def compute_scores_nsqip(y_pred_prob, y):
    scores = {}

    roc = BinaryAUROC(thresholds=None)
    pr = BinaryAveragePrecision(thresholds=None)


    roc_score = roc(y_pred_prob[:,-1], y)
    pr_score = pr(y_pred_prob[:,-1], y)
    


    scores["roc"] = roc_score
    scores["pr"] = pr_score
    return scores
def compute_scores_uci_cdc(y_pred_prob, y):
    scores = {}

    roc = BinaryAUROC(thresholds=None)
    pr = BinaryAveragePrecision(thresholds=None)


    roc_score = roc(y_pred_prob[:,-1], y)
    pr_score = pr(y_pred_prob[:,-1], y)
    
    acc = accuracy_score(torch.argmax(y_pred_prob, dim  = 1), y)
    scores["acc"] = acc

    scores["roc"] = roc_score
    scores["pr"] = pr_score
    return scores
def compute_scores_mover(y_pred_prob, y):
    scores = {}


    roc = BinaryAUROC(thresholds=None)
    pr = BinaryAveragePrecision(thresholds=None)


    roc_score = roc(y_pred_prob[:,-1], y)
    pr_score = pr(y_pred_prob[:,-1], y)
    

    scores["roc"] = roc_score
    scores["pr"] = pr_score
    return scores
