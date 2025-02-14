from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
import os
import argparse

current_dir = os.getcwd()
# Get the grandparent directory using a relative path
grandparent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
# Add the grandparent directory to the module search path
sys.path.append(grandparent_dir)

from data.uci.uci_dataset import uci_dat
from sklearn.ensemble import RandomForestClassifier
import torch
from training_func.exp_fnc import train_teacher, train_mlp_baseline
from data.create_dat import create_baseline_dataset, create_pretraining_dataset, create_all_datasets
from training_func.train_ssl import fine_tune_SSL, fine_tune_DK_SP, pretrain_SSL_baseline, fine_tune_PL_SSL, fine_tune_LUPISSL
from training_func.train_priv_ssl import pretrain_PIReg
from training_func.LUPI import KD_baseline, train_tram, train_twosteps


# Parse command-line arguments
parser = argparse.ArgumentParser(description='UCI Experiment')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--dataset', type=str, default='uci_cdc', help='Dataset name')
parser.add_argument('--cuda_device', type=str, default='0', help='CUDA device number')
# cuda_device = sys.argv[3]



args = parser.parse_args()
seed = int(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)



if args.dataset == "uci_cdc":
    # fetch dataset 

    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    
    # data (as pandas dataframes) 
    X = cdc_diabetes_health_indicators.data.features 
    y = cdc_diabetes_health_indicators.data.targets 
    correlation_matrix = np.corrcoef(X.T, y.T)[-1, :-1]


    priv_fea_indices = np.abs(correlation_matrix)<0.7
    regular_fea_indices = np.abs(correlation_matrix)<0.2

    x_priv_feas = X.columns[priv_fea_indices]
    x_reg_feas = X.columns[regular_fea_indices]
    split = (0.05,0.1,0.05,0.8)

elif args.dataset == "uci_iot_UDP":
    rt_iot2022 = fetch_ucirepo(id=942) 
    
    # data (as pandas dataframes) 
    X = rt_iot2022.data.features 
    y = rt_iot2022.data.targets 

    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numerical_columns]
    encoded_df = pd.get_dummies(y)
    # y = encoded_df['Attack_type_Thing_Speak']
    y = encoded_df['Attack_type_NMAP_UDP_SCAN']


    correlation_matrix = np.corrcoef(X.T, y.T)[-1, :-1]

    priv_fea_indices = np.abs(correlation_matrix)<0.7
    regular_fea_indices = np.abs(correlation_matrix)<0.15

    x_priv_feas = X.columns[priv_fea_indices]
    x_reg_feas = X.columns[regular_fea_indices]
    split =  (0.05,0.05,0.05,0.85)
elif args.dataset == "uci_iot_DOS":
    rt_iot2022 = fetch_ucirepo(id=942) 
    
    # data (as pandas dataframes) 
    X = rt_iot2022.data.features 
    y = rt_iot2022.data.targets 

    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numerical_columns]
    encoded_df = pd.get_dummies(y)
    # y = encoded_df['Attack_type_Thing_Speak']
    y = encoded_df['Attack_type_DOS_SYN_Hping']

    correlation_matrix = np.corrcoef(X.T, y.T)[-1, :-1]

    priv_fea_indices = np.abs(correlation_matrix)<0.7
    regular_fea_indices = np.abs(correlation_matrix)<0.15


    x_priv_feas = X.columns[priv_fea_indices]
    x_reg_feas = X.columns[regular_fea_indices]
    split =  (0.05,0.05,0.05,0.85)
elif args.dataset == "uci_iot_Speak":
    rt_iot2022 = fetch_ucirepo(id=942) 
    
    # data (as pandas dataframes) 
    X = rt_iot2022.data.features 
    y = rt_iot2022.data.targets 

    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numerical_columns]
    encoded_df = pd.get_dummies(y)
    y = encoded_df['Attack_type_Thing_Speak']
    # y = encoded_df['Attack_type_DOS_SYN_Hping']
    correlation_matrix = np.corrcoef(X.T, y.T)[-1, :-1]

    priv_fea_indices = np.abs(correlation_matrix)<0.7
    regular_fea_indices = np.abs(correlation_matrix)<0.15

    x_priv_feas = X.columns[priv_fea_indices]
    x_reg_feas = X.columns[regular_fea_indices]
    split =  (0.05,0.05,0.05,0.85)
elif args.dataset == "uci_phishing":
    # fetch dataset 
    phiusiil_phishing_url_website = fetch_ucirepo(id=967) 
    
    # data (as pandas dataframes) 
    X = phiusiil_phishing_url_website.data.features 
    y = phiusiil_phishing_url_website.data.targets 

    # Find all columns with numerical values in X
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numerical_columns]

    correlation_matrix = np.corrcoef(X.T, y.T)[-1, :-1]
        
    priv_fea_indices = np.abs(correlation_matrix)<0.7
    regular_fea_indices = np.abs(correlation_matrix)<0.2
    x_priv_feas = X.columns[priv_fea_indices]
    x_reg_feas = X.columns[regular_fea_indices]
    # split = (0.1,0.1,0.05,0.75)
    # split = (0.01,0.05,0.01,0.93)
    # split = (0.05,0.05,0.05,0.85)
    split = (0.025,0.05,0.025,0.9)
else:
    print("Dataset not available")
    sys.exit(1)



dat = uci_dat(X, y,  seed, split, x_priv_feas, x_reg_feas)


hpara = {}


"""====================dataset related setting=================="""
hpara["dataset"] = "uci" 
hpara["use_proj"] = True

hpara["batch_size"] = 128
hpara["num_workers"] = 0



"""====================Model related setting=================="""
# Dimensions
hpara["d_hid"] = 256
hpara["d_h"] = 128
hpara["d_proj"] = 128
hpara["d_out"] = 2

hpara["d_reg"] = dat.x_reg_tr.shape[-1]
hpara["d_priv"] = dat.x_priv_tr.shape[-1]

# how much data to corrupt to generate views, used by scarf
hpara["corruption_rate"] = 0.3
hpara["gamma"] = {"GenD": 1.0, "KD_SP":  1.0}
hpara["T"] = {"GenD": 1.0, "KD_SP":  1.0}

hpara["num_layers_enc"] = 3
hpara["num_layers_clf"] = 3
hpara["num_layers_proj"] = 3

hpara["corruption_scarf"] = 0.3
hpara["vime_corruption"] = 0.3
hpara["vime_alpha"] = 1.0


hpara["lfr_target_sample_ratio"] = 5
hpara["lfr_lambd"] = 0.01
hpara["lfr_num_targets"] = 8


hpara["infomax_la_R"] = 0.01
hpara["infomax_la_mu"] = 0.01
hpara["infomax_R_eps_weight"] = 1e-08

"""====================Optimization related setting=================="""

hpara["early_stopping"] = True; hpara["patience"] = 3; 

hpara["smooth_sim"] = True


"""====================dataset related setting=================="""
hpara["split"] = split

stop_gradient = False

device = "cuda" if torch.cuda.is_available() else "cpu"


hpara["class_weights"] = None
hpara["seed"] = seed
hpara["Reg_min"] = dat.stats["Reg_min"] 
hpara["Reg_max"] = dat.stats["Reg_max"] 
hpara["Priv_min"] = dat.stats["Priv_min"] 
hpara["Priv_max"] = dat.stats["Priv_max"] 



D_tr, D_val, D_test = create_baseline_dataset(hpara, dat)
D_tr_pre = create_pretraining_dataset(hpara, dat)

for i in D_tr:
    hpara["sample_data"] = i[0]
    break

result = {}
model_result = {}
proj_path = "PATH TO PROJECT"



save_path = proj_path + "/results/{}".format(args.dataset )

if not os.path.exists(proj_path + "/results"):
    os.makedirs(proj_path + "/results")

if not os.path.exists(save_path):
    os.makedirs(save_path)


ft_data ={"tr": D_tr,  "val": D_val, "tst": D_test, "pretraining": D_tr_pre}

hpara["lr"]  = 1e-4
hpara["max_epoch"] = 200

backbone_name  = "corinfomax"
ssl_pt_teacher, ssl_ft, _, _, _, _, _ = pretrain_SSL_baseline(hpara,  dat.stats, ft_data, backbone = backbone_name, mode="Priv")
result, ssl_teacher = fine_tune_SSL(hpara, result, ssl_ft, ft_data, stop_gradient = stop_gradient, mode="Priv", model_name = backbone_name + " teacher")


teacher_model = ssl_teacher
ft_data = create_all_datasets(hpara, dat, teacher_model,ft_data, teacher_mode  ="GenD")


for lr in [1e-2,1e-3, 1e-4]:
    hpara["lr"]  = lr
    result, mlp_baseline = train_mlp_baseline(result, hpara, D_tr, D_val, D_test)

    result, tram_baseline = train_twosteps(result, hpara, teacher_model, D_tr, D_val, D_test)

    result, GenD_model = KD_baseline(hpara, result, ft_data)

    
    result, tram_baseline = train_tram(result, hpara, D_tr, D_val, D_test)

    for backbone_name in ["priv_corinfomax", "priv_vime", "TriDeNT"]:
        try:
            hpara["loss_comp"] = ["proj", "closs", "dloss" ,  "priv_cov_loss"]
            ssl_pt, ssl_ft, ssl_ft_GenD, ssl_ft_PFD, ssl_ft_PL, ssl_ft_Semi_GenD, ssl_ft_Semi_PFD = pretrain_PIReg(hpara, ft_data, backbone = backbone_name, teacher= teacher_model)
            result, ssl_baseline = fine_tune_SSL(hpara, result, ssl_ft, ft_data, stop_gradient = stop_gradient, mode="Reg", model_name = backbone_name + " PIReg")
            result, ssl_ft_PL = fine_tune_PL_SSL(hpara, result, ssl_ft_PL, ft_data, stop_gradient = stop_gradient, model_name = backbone_name+ " PIReg")
            result,  ssl_ft_GenD = fine_tune_LUPISSL(hpara, result, ssl_ft_GenD, ft_data, stop_gradient = stop_gradient, model_name = backbone_name+ " PIReg")
            result,  ssl_ft_DK_SP = fine_tune_DK_SP(hpara, result, ssl_ft_Semi_GenD, ft_data, stop_gradient = stop_gradient, model_name = backbone_name+ " PIReg")
        except:
            print("Model {} not available".format(backbone_name))
            continue


    for backbone_name in ["corinfomax", "simsiam", "vime","barlow", "vicreg", "scarf"]:
        try:
            ssl_pt, ssl_ft, ssl_ft_GenD, ssl_ft_PFD, ssl_ft_PL, ssl_ft_Semi_GenD, ssl_ft_Semi_PFD = pretrain_SSL_baseline(hpara,  dat.stats, ft_data, backbone = backbone_name, mode="Reg")
            result, ssl_baseline = fine_tune_SSL(hpara, result, ssl_ft, ft_data, stop_gradient = stop_gradient, mode="Reg", model_name = backbone_name)
            result, ssl_ft_PL  = fine_tune_PL_SSL(hpara, result, ssl_ft_PL, ft_data, stop_gradient = stop_gradient, model_name = backbone_name)
            result,  ssl_ft_GenD = fine_tune_LUPISSL(hpara, result,  ssl_ft_GenD, ft_data, stop_gradient = stop_gradient, model_name = backbone_name)
        except:
            print("Model {} not available".format(backbone_name))
            continue

    torch.save(result, save_path+"/{}_{}_{}_{}_{}_{}.pt".format(split[0], split[1], split[2], split[3], seed, lr))



