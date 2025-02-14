
import torch
# sys.path.insert(0, '..')
from data.create_dat import database
from data.mnist.dataset_split import return_mnist_split
from sklearn.model_selection import train_test_split


def return_input(dat_path, trans_name):
    x_tr = torch.load(dat_path + '/processed/{}/train.dat'.format(trans_name))
    x_test = torch.load(dat_path + '/processed/{}/test.dat'.format(trans_name))
    return x_tr, x_test


def split_data(X, y, tr_ratio, tr_ratio_unlabel, val_ratio, test_ratio, random_state):
    if tr_ratio + tr_ratio_unlabel + val_ratio + test_ratio != 1.0:
        raise ValueError("The sum of split ratios must be 1.0")
    
    # Split into initial training and temp sets
    X_train_labeled, X_temp, y_train_labeled, y_temp = train_test_split(X, y, test_size=1 - tr_ratio - tr_ratio_unlabel, random_state=random_state)
    
    # Further split temp into validation and test sets
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_test_ratio, random_state=random_state)

    # Split labeled training set into labeled and unlabeled subsets
    unlabeled_ratio = tr_ratio_unlabel / (tr_ratio + tr_ratio_unlabel)
    X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(
        X_train_labeled, y_train_labeled, test_size=unlabeled_ratio, random_state=random_state)
    assert len(set(X_train_labeled.index) & set(X_train_unlabeled.index) & set(X_val.index)& set(X_test.index)) == 0
    
    return X_train_labeled, X_train_unlabeled, X_val, X_test, y_train_labeled, y_train_unlabeled, y_val, y_test



class uci_dat(database):
    def __init__(self, X, y,  seed, split, x_priv_feas, x_reg_feas):
        tr_ratio, tr_ratio_unlabel, val_ratio, test_ratio = split

        X_train_labeled, X_train_unlabeled, X_val, X_test, y_train_labeled, y_train_unlabeled, y_val, y_test = split_data(X, y, tr_ratio, tr_ratio_unlabel, val_ratio, test_ratio, random_state=seed)

        x_reg_tr = X_train_labeled[x_reg_feas]
        x_reg_val = X_val[x_reg_feas]
        x_reg_un = X_train_unlabeled[x_reg_feas]
        x_test = X_test[x_reg_feas]
        

        

        x_priv_tr = X_train_labeled[x_priv_feas]
        x_priv_val = X_val[x_priv_feas]
        x_priv_un = X_train_unlabeled[x_priv_feas]
        x_priv_test = X_test[x_priv_feas]


        self.x_reg_tr = torch.from_numpy(x_reg_tr.values).float()
        self.x_priv_tr = torch.from_numpy(x_priv_tr.values).float()
        self.labels_tr = torch.from_numpy(y_train_labeled.values).long().flatten()

        self.x_reg_val = torch.from_numpy(x_reg_val.values).float()
        self.x_priv_val = torch.from_numpy(x_priv_val.values).float()
        self.labels_val = torch.from_numpy(y_val.values).long().flatten()

        
        self.x_reg_tst = torch.from_numpy(x_test.values).float()
        self.x_priv_tst = torch.from_numpy(x_priv_test.values).float()
        self.labels_tst = torch.from_numpy(y_test.values).long().flatten()
        
        self.x_reg_un = torch.from_numpy(x_reg_un.values).float()
        self.x_priv_un = torch.from_numpy(x_priv_un.values).float()

        self.get_stats()



