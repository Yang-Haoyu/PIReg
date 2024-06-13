
import torch
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np




def create_mnist_split(dat_path, split=(1000, 1000, 50000)):
    x_reg_tr= torch.load(dat_path + '/processed/Resize/train.dat')

    tr_size, val_size, unlabeled_size = split

    used_idx = list(range(len(x_reg_tr)))

    tr_val_idx, unlabeled_idx = train_test_split(used_idx, test_size=unlabeled_size, random_state=0)

    for seed in range(10):
        tmp_idx, val_idx = train_test_split(tr_val_idx, test_size=val_size, random_state=seed)
        drop_idx, tr_idx = train_test_split(tmp_idx, test_size=tr_size, random_state=seed)

        assert sorted(used_idx) == sorted(drop_idx + tr_idx + val_idx + unlabeled_idx)



        df_tr = pd.DataFrame(tr_idx)
        df_tr.to_csv(dat_path + "/cv_unabeled_{}_{}_{}/tr_idx_{}.csv".format(tr_size, val_size, unlabeled_size, seed))
        df_val = pd.DataFrame(val_idx)
        df_val.to_csv(dat_path + "/cv_unabeled_{}_{}_{}/val_idx_{}.csv".format(tr_size, val_size, unlabeled_size, seed))
        df_un = pd.DataFrame(unlabeled_idx)
        df_un.to_csv(dat_path + "/cv_unabeled_{}_{}_{}/unlabeled_idx_{}.csv".format(tr_size, val_size, unlabeled_size, seed))

def return_mnist_split(dat_path, seed=0, split=(5000, 5000, 50000)):
    tr_size, val_size, unlabeled_size = split
    
    if not os.path.exists(dat_path + "/cv_unabeled_{}_{}_{}".format(tr_size, val_size, unlabeled_size)):
        os.mkdir(dat_path +  "/cv_unabeled_{}_{}_{}".format(tr_size, val_size, unlabeled_size))

    if not os.path.exists(dat_path + "/cv_unabeled_{}_{}_{}/tr_idx_{}.csv".format(tr_size, val_size, unlabeled_size, seed)):

        print("Didn't find existing split, create a new one")
        create_mnist_split(dat_path, split=split)

    df_tr = pd.read_csv(dat_path + "/cv_unabeled_{}_{}_{}/tr_idx_{}.csv".format(tr_size, val_size, unlabeled_size, seed), index_col=0)
    df_val = pd.read_csv(dat_path + "/cv_unabeled_{}_{}_{}/val_idx_{}.csv".format(tr_size, val_size, unlabeled_size, seed), index_col=0)
    df_un = pd.read_csv(dat_path + "/cv_unabeled_{}_{}_{}/unlabeled_idx_{}.csv".format(tr_size, val_size, unlabeled_size, seed), index_col=0)
    return np.concatenate(df_tr[:tr_size].values), np.concatenate(df_val.values), np.concatenate(df_un.values)



if __name__ == "__main__":
    dat_path = "PATH TO MNOST"
    tr_idx, val_idx, un_idx = return_mnist_split(dat_path, seed=0, split=(1000, 1000, 5000))