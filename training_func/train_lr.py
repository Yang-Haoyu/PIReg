
from sklearn.linear_model import LogisticRegression
import torch
from  torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryAveragePrecision
D_tr, D_val, D_test

x_priv = []
x = []
y = []
for i in D_tr:
    x.append(i[0])
    x_priv.append(i[1])
    y.append(i[2])
for i in D_val:
    x.append(i[0])
    x_priv.append(i[1])
    y.append(i[2])

x = torch.concat(x)
x_priv = torch.concat(x_priv)
y = torch.concat(y)


clf = LogisticRegression()
clf.fit(x, y)


for i in D_test:
    x_tst = i[0]
    x_priv_tst = i[1]
    y_tst = i[2]

y_pred_prob = torch.from_numpy(clf.predict_proba(x_tst)[:, 1])


roc = BinaryAUROC(thresholds=None)
pr = BinaryAveragePrecision(thresholds=None)


roc_score = roc(y_pred_prob, y_tst)
pr_score = pr(y_pred_prob, y_tst)


scores["roc"] = roc_score
scores["pr"] = pr_score