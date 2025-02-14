
import lightning as L
import torch


from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from methods.MLP import clf_mlp


def train_models(model, model_name, hpara, data_tr, data_val):
    
    early_stopping = hpara["early_stopping"]
    patience = hpara["patience"]

    max_epoch = hpara["max_epoch"]


    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    _logger = CSVLogger("logs", name= model_name)

    if early_stopping:
        
        monitor = "val_loss"
        mode = "min"
        callbacks = [EarlyStopping(monitor=monitor, mode=mode, patience=patience)]
    else:
        callbacks = None

    # train
    trainer = L.Trainer(accelerator = accelerator, max_epochs=max_epoch, enable_checkpointing=False, deterministic=True, logger=_logger, callbacks = callbacks)
    trainer.fit(model=model, train_dataloaders=data_tr, val_dataloaders=data_val)

    return model, trainer



def train_teacher(result, hpara, data_tr, data_val, data_test, mode):
    if mode == "GenD":
        # GenD Teacher
        mlp_clf_gend = clf_mlp(hpara["d_priv"], hpara, mode = "GenD")
        teacher, trainer_gend = train_models(mlp_clf_gend, "Teacher GenD", hpara, data_tr, data_val)
        result["Teacher GenD"] = trainer_gend.test(teacher, data_test)[0]
        # result["Teacher GenD"]["val_loss"] = trainer_gend.test(teacher, data_val)[0]
    elif mode == "PFD":
        # PFD Teacher
        mlp_clf_pfd = clf_mlp(hpara["d_priv"] + hpara["d_reg"], hpara,  mode = "PFD")
        teacher, trainer_pfd = train_models(mlp_clf_pfd, "Teacher PFD", hpara, data_tr, data_val)
        result["Teacher PFD"] = trainer_pfd.test(teacher, data_test)[0]
        # result["Teacher PFD"]["val_loss"] = trainer_pfd.test(teacher, data_val)[0]
        
    else:
        raise NotImplementedError

    return result, teacher


def train_mlp_baseline(result, hpara, data_tr, data_val, data_test):

    clf_baseline = clf_mlp(hpara["d_reg"], hpara, mode = "Reg")
    clf_baseline, trainer_clf_baseline = train_models(clf_baseline, "MLP", hpara, data_tr, data_val )

    result["MLP"] = trainer_clf_baseline.test(clf_baseline, data_test)[0]
    # result["MLP"]["val_loss"] = trainer_clf_baseline.test(clf_baseline, data_val)[0]

    return result, clf_baseline

