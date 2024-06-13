
import torch
# sys.path.insert(0, '..')
from data.create_dat import database
from data.mnist.dataset_split import return_mnist_split



def return_input(dat_path, trans_name):
    x_tr = torch.load(dat_path + '/processed/{}/train.dat'.format(trans_name))
    x_test = torch.load(dat_path + '/processed/{}/test.dat'.format(trans_name))
    return x_tr, x_test

class mnist_priv(database):
    def __init__(self, dat_path,  seed, split, reg_trans = "Blur",  priv_trans = "Resize"):

        

        x_reg_tr, x_test = return_input(dat_path, reg_trans)
        x_priv_tr, x_priv_test = return_input(dat_path, priv_trans)

        y_tr= torch.load(dat_path + '/processed/tr_label.dat').long()
        y_test = torch.load(dat_path + '/processed/tst_label.dat').long()



        tr_idx, val_idx, un_idx = return_mnist_split(dat_path, seed=seed, split=split)


        self.x_reg_tr = x_reg_tr[tr_idx]
        self.x_priv_tr = x_priv_tr[tr_idx]
        self.labels_tr = y_tr[tr_idx]

        self.x_reg_val = x_reg_tr[val_idx]
        self.x_priv_val = x_priv_tr[val_idx]
        self.labels_val = y_tr[val_idx]

        
        self.x_reg_tst = x_test
        self.x_priv_tst = x_priv_test
        self.labels_tst = y_test
        
        self.x_reg_un = x_reg_tr[un_idx]
        self.x_priv_un = x_priv_tr[un_idx]

        self.get_stats()



