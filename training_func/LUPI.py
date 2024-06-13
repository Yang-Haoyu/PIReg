from methods.LUPI_kd import LUPI_KD
from training_func.exp_fnc import train_models
from methods.tram import tram
from methods.lupts import twosteps


def train_twosteps(result, hpara, teacher, data_tr, data_val, data_test):

    clf_baseline = twosteps(hpara, teacher)
    clf_baseline, trainer_clf_baseline = train_models(clf_baseline, "twosteps", hpara, data_tr, data_val )

    result["twosteps"] = trainer_clf_baseline.test(clf_baseline, data_test)[0]
    result["twosteps"]["val_loss"] = trainer_clf_baseline.test(clf_baseline, data_val)[0]

    return result, clf_baseline

def train_tram(result, hpara, data_tr, data_val, data_test):

    clf_baseline = tram(hpara)
    clf_baseline, trainer_clf_baseline = train_models(clf_baseline, "TRAM", hpara, data_tr, data_val )

    result["TRAM"] = trainer_clf_baseline.test(clf_baseline, data_test)[0]
    result["TRAM"]["val_loss"] = trainer_clf_baseline.test(clf_baseline, data_val)[0]

    return result, clf_baseline
def Semi_KD_baseline(hpara, result, ft_data):



    "------------------------------------------------ Regular learning ----------------------------------------------------"
    GenD_model = LUPI_KD(hpara["d_reg"],  hpara, gamma =  hpara["gamma"]["GenD"], T=hpara["T"]["GenD"],
                         encoder = None, stop_gradient = False)

    GenD_model, trainer_GenD =  train_models(GenD_model, "GenD", hpara, ft_data["GenD_tr"], ft_data["GenD_val"])
    result["GenD"] = trainer_GenD.test(GenD_model, ft_data["tst"])[0]
    result["GenD"]["val_loss"] = trainer_GenD.test(GenD_model, ft_data["val"])[0]


    PFD_model = LUPI_KD(hpara["d_reg"], hpara, gamma =  hpara["gamma"]["PFD"], T=hpara["T"]["PFD"],
                    encoder = None, stop_gradient = False)

    PFD_model, trainer_PFD =  train_models(PFD_model, "PFD", hpara, ft_data["PFD_tr"], ft_data["PFD_val"])
    result["PFD"] = trainer_PFD.test(PFD_model, ft_data["tst"])[0]
    result["PFD"]["val_loss"] = trainer_PFD.test(PFD_model, ft_data["val"])[0]


    "------------------------------------------------ semi learning ----------------------------------------------------"
    # Fine Tunining using KD
    Semi_GenD = LUPI_KD(hpara["d_reg"],  hpara, gamma =  hpara["gamma"]["Semi GenD"], T=hpara["T"]["Semi GenD"],
                    encoder = None, stop_gradient = False)

    Semi_GenD, trainer_Semi_GenD =  train_models(Semi_GenD, "Semi GenD", hpara, ft_data["Semi_GenD_tr"], ft_data["GenD_val"])
    result["Semi GenD"] = trainer_Semi_GenD.test(Semi_GenD, ft_data["tst"])[0]
    result["Semi GenD"]["val_loss"] = trainer_Semi_GenD.test(Semi_GenD, ft_data["val"])[0]




    # Fine Tunining using KD
    Semi_PFD = LUPI_KD(hpara["d_reg"], hpara, gamma =  hpara["gamma"]["Semi PFD"], T=hpara["T"]["Semi PFD"],
                    encoder = None, stop_gradient = False)

    Semi_PFD, trainer_Semi_PFD =  train_models(Semi_PFD, "Semi PFD", hpara, ft_data["Semi_PFD_tr"], ft_data["PFD_val"])
    result["Semi PFD"] = trainer_Semi_PFD.test(Semi_PFD, ft_data["tst"])[0]
    result["Semi PFD"]["val_loss"] = trainer_Semi_PFD.test(Semi_PFD, ft_data["val"])[0]

    return result, GenD_model, PFD_model, Semi_GenD, Semi_PFD
