
import sys
import os

current_dir = os.getcwd()
# Get the grandparent directory using a relative path
grandparent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
# Add the grandparent directory to the module search path
sys.path.append(grandparent_dir)

from data.mnist.dataset import mnist_priv

import torch


from training_func.exp_fnc import train_teacher, train_mlp_baseline
from data.create_dat import create_baseline_dataset, create_pretraining_dataset, create_all_datasets
from training_func.train_ssl import fine_tune_SSL, fine_tune_SemiSSL, pretrain_SSL_baseline, fine_tune_PL_SSL, fine_tune_LUPISSL
from training_func.train_priv_ssl import pretrain_priv_SSL
from training_func.LUPI import Semi_KD_baseline, train_tram, train_twosteps

save_path = grandparent_dir + "/mnist_result"
if not os.path.exists(save_path):
    os.mkdir(save_path)

# dat_path = "PATH TO MNIST"
dat_path = sys.argv[1]
seed = int(sys.argv[2])


for priv_trans in ["Resize"]:

    for reg_trans  in ["Blur"]:
        for split in [(1000, 1000, 5000), (2500, 1000, 5000), (5000, 1000, 5000)]:

            
            # reg_trans, priv_trans = "Blur", "Elastic"
            if reg_trans == priv_trans:
                continue
            hpara = {}


            """====================dataset related setting=================="""
            hpara["dataset"] = "mnist" 
            hpara["use_proj"] = True

            hpara["batch_size"] = 512
            hpara["num_workers"] = 0
            hpara["dat_path"] = dat_path

            #  ["Blur","Elastic","Resize", "Affine"]
            hpara["reg_trans"], hpara["priv_trans"] = reg_trans, priv_trans
            """====================Model related setting=================="""
            # Dimensions
            hpara["d_hid"] = 256
            hpara["d_h"] = 128
            hpara["d_proj"] = 128
            hpara["d_out"] = 10
            priv_size=25
            reg_size=25
            hpara["d_reg"] = priv_size**2
            hpara["d_priv"] = reg_size**2

            # how much data to corrupt to generate views, used by scarf
            hpara["corruption_rate"] = 0.3
            hpara["gamma"] = {"GenD": 1.0, "KD_SP":  0.8}
            hpara["T"] = {"GenD": 0.1, "KD_SP":  10.0}

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

            hpara["early_stopping"] = True; hpara["patience"] = 5; hpara["max_epoch"] = 750

            hpara["smooth_sim"] = True


            """====================dataset related setting=================="""
            hpara["split"] = split

            stop_gradient = False

            device = "cuda" if torch.cuda.is_available() else "cpu"


            hpara["class_weights"] = None
            hpara["seed"] = seed


            hpara["lr"]  = 1e-3

            dat = mnist_priv( dat_path,  seed, split, reg_trans = hpara["reg_trans"],  priv_trans = hpara["priv_trans"])



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

            result, gend_teacher = train_teacher(result, hpara, D_tr, D_val, D_test, "GenD")

            ft_data ={"tr": D_tr,  "val": D_val, "tst": D_test, "pretraining": D_tr_pre}

            backbone_name  = "corinfomax"
            ssl_pt_teacher, ssl_ft, _, _, _, _, _ = pretrain_SSL_baseline(hpara,  dat.stats, ft_data, backbone = backbone_name, mode="Priv")
            result, ssl_teacher = fine_tune_SSL(hpara, result, ssl_ft, ft_data, stop_gradient = stop_gradient, mode="Priv", model_name = backbone_name + " teacher")


            teacher_model = ssl_teacher
            ft_data = create_all_datasets(hpara, dat, teacher_model,ft_data)


            for lr in [1e-4]:
                hpara["lr"]  = lr
                result, tram_baseline = train_twosteps(result, hpara, teacher_model, D_tr, D_val, D_test)

                result, mlp_baseline = train_mlp_baseline(result, hpara, D_tr, D_val, D_test)
                result, tram_baseline = train_tram(result, hpara, D_tr, D_val, D_test)


                result, GenD_model, Semi_GenD = Semi_KD_baseline(hpara, result, ft_data)
                hpara["loss_corinfomax"] = "cross"
                for backbone_name in ["TriDeNT", "priv_vime2"]:
                    ssl_pt, ssl_ft, ssl_ft_GenD, ssl_ft_PFD, ssl_ft_PL, ssl_ft_Semi_GenD, ssl_ft_Semi_PFD = pretrain_priv_SSL(hpara, ft_data, backbone = backbone_name, teacher= teacher_model)
                    result, ssl_baseline = fine_tune_SSL(hpara, result, ssl_ft, ft_data, stop_gradient = stop_gradient, mode="Reg", model_name = backbone_name + " privssl")
                    result, ssl_ft_PL = fine_tune_PL_SSL(hpara, result, ssl_ft_PL, ft_data, stop_gradient = stop_gradient, model_name = backbone_name+ " privssl")
                    result,  ssl_ft_GenD = fine_tune_LUPISSL(hpara, result, (ssl_ft_GenD, ssl_ft_PFD), ft_data, stop_gradient = stop_gradient, model_name = backbone_name+ " privssl")
                    result,  ssl_ft_DK_ = fine_tune_SemiSSL(hpara, result, (ssl_ft_Semi_GenD, ssl_ft_Semi_PFD), ft_data, stop_gradient = stop_gradient, model_name = backbone_name+ " privssl")



                for backbone_name in ["corinfomax", "simsiam", "vime","barlow", "vicreg", "scarf"]:

                    ssl_pt, ssl_ft, ssl_ft_GenD, ssl_ft_PFD, ssl_ft_PL, ssl_ft_Semi_GenD, ssl_ft_Semi_PFD = pretrain_SSL_baseline(hpara,  dat.stats, ft_data, backbone = backbone_name, mode="Reg")
                    result, ssl_baseline = fine_tune_SSL(hpara, result, ssl_ft, ft_data, stop_gradient = stop_gradient, mode="Reg", model_name = backbone_name)
                    result, ssl_ft_PL  = fine_tune_PL_SSL(hpara, result, ssl_ft_PL, ft_data, stop_gradient = stop_gradient, model_name = backbone_name)
                    result,  ( ssl_ft_GenD, ssl_ft_PFD) = fine_tune_LUPISSL(hpara, result,  (ssl_ft_GenD, ssl_ft_PFD), ft_data, stop_gradient = stop_gradient, model_name = backbone_name)

                torch.save(result, save_path+"/{}_{}_{}_{}_{}_{}_{}.pt".format(reg_trans, priv_trans, split[0], split[1], split[2], seed, lr))



