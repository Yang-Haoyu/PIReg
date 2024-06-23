from training_func.exp_fnc import train_models

from priv_reg.PIReg import pcorinfomax, pvime
from priv_reg.TriDeNT import TriDeNT
from priv_reg.priv_ssl import priv_ssl

def create_PIReg(hpara, backbone = "scarf", teacher = None):
        # make three copies of pretrained model for different fine tuning strategy

    if backbone == "priv_corinfomax":
        ssl = pcorinfomax(hpara, teacher=teacher)
    elif backbone == "priv_vime":
        ssl = pvime(hpara, teacher=teacher)
    elif backbone == "TriDeNT":
        ssl = TriDeNT(hpara, teacher=teacher)
    else:
        print("Please choose backbone from [scarf, vime]")
        raise NotImplementedError
    return ssl


def pretrain_PIReg(hpara, ft_data, backbone = None, teacher = None):
    # dataset=hpara["dataset"]
    D_tr_pre, D_val = ft_data["pretraining"], ft_data["val"]
    # make three copies of pretrained model for different fine tuning strategy
    ssl_pt = create_PIReg(hpara, backbone = backbone, teacher = teacher)


    # -------------- Pretraining ---------------------------
    ssl = priv_ssl(ssl_pt, hpara)

    ssl_pt, _ = train_models(ssl, "priv_ssl_priv_embd", hpara, D_tr_pre, D_val)

    ssl_ft = create_PIReg( hpara, backbone = backbone, teacher = teacher)
    ssl_ft_PL = create_PIReg( hpara, backbone = backbone, teacher = teacher)
    ssl_ft_GenD = create_PIReg( hpara, backbone = backbone, teacher = teacher)
    ssl_ft_PFD = create_PIReg( hpara,  backbone = backbone, teacher = teacher)
    ssl_ft_Semi_GenD = create_PIReg( hpara, backbone = backbone, teacher = teacher)
    ssl_ft_Semi_PFD = create_PIReg( hpara,  backbone = backbone, teacher = teacher)

    ssl_ft.load_state_dict(ssl_pt.ssl_backbone.state_dict())
    ssl_ft_GenD.load_state_dict(ssl_pt.ssl_backbone.state_dict())
    ssl_ft_PFD.load_state_dict(ssl_pt.ssl_backbone.state_dict())
    ssl_ft_PL.load_state_dict(ssl_pt.ssl_backbone.state_dict())
    ssl_ft_Semi_GenD.load_state_dict(ssl_pt.ssl_backbone.state_dict())
    ssl_ft_Semi_PFD.load_state_dict(ssl_pt.ssl_backbone.state_dict())

    return ssl_pt, ssl_ft, ssl_ft_GenD, ssl_ft_PFD, ssl_ft_PL, ssl_ft_Semi_GenD, ssl_ft_Semi_PFD

