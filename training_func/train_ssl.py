from training_func.exp_fnc import train_models

from methods.vime import vime
from methods.lfr import lfr
from methods.siams import vicreg, simsiam, barlowtwins, SCARF, corinfomax

from methods.MLP import clf_mlp
from methods.LUPI_kd import LUPI_KD


def create_ssl_models(d_in, hpara, stats, backbone = "scarf", mode = "priv"):
        # make three copies of pretrained model for different fine tuning strategy
    if backbone == "scarf":
        ssl = SCARF(d_in, hpara, stats['{}_min'.format(mode)], stats['{}_max'.format(mode)], mode = mode)
    elif backbone == "vicreg":
        ssl = vicreg(d_in, hpara, stats['{}_min'.format(mode)], stats['{}_max'.format(mode)], mode = mode)
    elif backbone == "corinfomax":
        ssl = corinfomax(d_in, hpara, stats['{}_min'.format(mode)], stats['{}_max'.format(mode)], mode = mode)
    elif backbone == "barlow":
        ssl = barlowtwins(d_in, hpara, stats['{}_min'.format(mode)], stats['{}_max'.format(mode)], mode = mode)
    elif backbone == "simsiam":
        ssl = simsiam(d_in,  hpara, stats['{}_min'.format(mode)], stats['{}_max'.format(mode)], mode = mode)
    elif backbone == "vime":
        ssl = vime(d_in,  hpara, mode = mode)
    elif backbone == "lfr":
        ssl = lfr(d_in, hpara, mode = mode)

    else:
        print("Please choose backbone from [scarf, vime]")
        raise NotImplementedError
    return ssl


def pretrain_SSL_baseline(hpara, stats, ft_data, backbone = "scarf", mode = "Reg"):
    # dataset=hpara["dataset"]
    D_tr_pre, D_val = ft_data["pretraining"], ft_data["val"]
    # make three copies of pretrained model for different fine tuning strategy
    if mode == "Reg":
        ssl_pt = create_ssl_models(hpara["d_reg"], hpara, stats, backbone = backbone, mode = mode)
    elif mode == "Priv":
        ssl_pt = create_ssl_models(hpara["d_priv"], hpara,  stats, backbone = backbone, mode = mode)
    else:
        raise NotImplementedError

    # -------------- Pretraining ---------------------------
    ssl_pt, _ = train_models(ssl_pt, "ssl_priv_embd", hpara, D_tr_pre, D_val)

    # -------------- create new model for fine tuning ---------------------------
    if mode == "Reg":
        ssl_ft = create_ssl_models(hpara["d_reg"], hpara, stats, backbone = backbone, mode = mode)
        ssl_ft_PL = create_ssl_models(hpara["d_reg"], hpara, stats, backbone = backbone, mode = mode)
        ssl_ft_GenD = create_ssl_models(hpara["d_reg"], hpara, stats, backbone = backbone, mode = mode)
        ssl_ft_PFD = create_ssl_models(hpara["d_reg"], hpara, stats, backbone = backbone, mode = mode)

        ssl_ft_Semi_GenD = create_ssl_models(hpara["d_reg"], hpara, stats, backbone = backbone, mode = mode)
        ssl_ft_Semi_PFD = create_ssl_models(hpara["d_reg"], hpara, stats, backbone = backbone, mode = mode)
    elif mode == "Priv":
        ssl_ft = create_ssl_models(hpara["d_priv"], hpara,  stats, backbone = backbone, mode = mode)
        ssl_ft_PL = create_ssl_models(hpara["d_priv"], hpara,  stats, backbone = backbone, mode = mode)
        ssl_ft_GenD = create_ssl_models(hpara["d_priv"], hpara,  stats, backbone = backbone, mode = mode)
        ssl_ft_PFD = create_ssl_models(hpara["d_priv"], hpara,  stats, backbone = backbone, mode = mode)

        ssl_ft_Semi_GenD = create_ssl_models(hpara["d_priv"], hpara,  stats, backbone = backbone, mode = mode)
        ssl_ft_Semi_PFD = create_ssl_models(hpara["d_priv"], hpara,  stats, backbone = backbone, mode = mode)
    else:
        raise NotImplementedError


    ssl_ft.load_state_dict(ssl_pt.state_dict())
    ssl_ft_GenD.load_state_dict(ssl_pt.state_dict())
    ssl_ft_PFD.load_state_dict(ssl_pt.state_dict())
    ssl_ft_PL.load_state_dict(ssl_pt.state_dict())

    ssl_ft_Semi_GenD.load_state_dict(ssl_pt.state_dict())
    ssl_ft_Semi_PFD.load_state_dict(ssl_pt.state_dict())
    return ssl_pt, ssl_ft, ssl_ft_GenD, ssl_ft_PFD, ssl_ft_PL, ssl_ft_Semi_GenD, ssl_ft_Semi_PFD


def fine_tune_SSL(hpara, result, ssl_ft, ft_data, stop_gradient = False, model_name = "scarf", mode = "Reg"):

    # --------------------- Fine Tunining using labeled data --------------------- 
    ft_baseline = clf_mlp(None, hpara, mode = mode, encoder = ssl_ft.encoder, stop_gradient = stop_gradient)
    ft_baseline, trainer_ft_baseline =  train_models(ft_baseline, "{}".format(model_name), hpara, ft_data["tr"], ft_data["val"])
    result["{} {}".format(model_name, mode)] = trainer_ft_baseline.test(ft_baseline, ft_data["tst"])[0]
    result["{} {}".format(model_name, mode)]["val_loss"] = trainer_ft_baseline.test(ft_baseline, ft_data["val"])[0]

    return result, ft_baseline

def fine_tune_DK_SP(hpara, result, ssl_ft_DK_SP, ft_data, stop_gradient = False, model_name = "scarf"):
    
    # --------------------- Fine Tunining using KD GenD ---------------------
    ft_DK_SP = LUPI_KD(None, hpara, gamma =  hpara["gamma"]["KD_SP"], T=hpara["T"]["KD_SP"],
                      encoder = ssl_ft_DK_SP.encoder, stop_gradient = stop_gradient)
    
    ft_DK_SP, trainer_ft_GenD =  train_models(ft_DK_SP, "{}".format(model_name), hpara, ft_data["KD_SP_tr"], ft_data["KD_SP_tr"])
    result["{} KD_SP".format(model_name)] = trainer_ft_GenD.test(ft_DK_SP, ft_data["tst"])[0]
    result["{} KD_SP".format(model_name)]["val_loss"] = trainer_ft_GenD.test(ft_DK_SP, ft_data["val"])[0]


    return result, ft_DK_SP

def fine_tune_LUPISSL(hpara, result, ssl_ft_GenD, ft_data, stop_gradient = False, model_name = "scarf"):

    # --------------------- Fine Tunining using KD GenD ---------------------
    ft_GenD = LUPI_KD(None, hpara, gamma =  hpara["gamma"]["GenD"], T=hpara["T"]["GenD"],
                      encoder = ssl_ft_GenD.encoder, stop_gradient = stop_gradient)
    
    ft_GenD, trainer_ft_GenD =  train_models(ft_GenD, "{}".format(model_name), hpara, ft_data["GenD_tr"], ft_data["GenD_val"])
    result["{} GenD".format(model_name)] = trainer_ft_GenD.test(ft_GenD, ft_data["tst"])[0]
    result["{} GenD".format(model_name)]["val_loss"] = trainer_ft_GenD.test(ft_GenD, ft_data["val"])[0]

    return result, ft_GenD


def fine_tune_PL_SSL(hpara, result, ssl_ft_PL, ft_data, stop_gradient = False, model_name = "scarf"):
    

    # --------------------- Fine Tunining using Pseudo Labeled data ---------------------
    ft_PL = clf_mlp(None, hpara,  mode = "Reg", encoder = ssl_ft_PL.encoder, stop_gradient = stop_gradient)
    ft_PL, trainer_ft_PL =  train_models(ft_PL, "{} PL".format(model_name), hpara, ft_data["KD_SP_tr"], ft_data["val"])
    result["{} PL".format(model_name)] = trainer_ft_PL.test(ft_PL, ft_data["tst"])[0]
    result["{} PL".format(model_name)]["val_loss"] = trainer_ft_PL.test(ft_PL, ft_data["val"])[0]


    return result, ft_PL