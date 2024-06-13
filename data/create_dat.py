from torch.utils.data import  DataLoader, Dataset
import torch



class database():
    def __init__(self):

        self.x_reg_tr = None
        self.x_priv_tr = None
        self.labels_tr = None

        self.x_reg_val = None
        self.x_priv_val = None
        self.labels_val = None
        
        self.x_reg_tst = None
        self.x_priv_tst = None
        self.labels_tst = None
        
        self.x_reg_un = None
        self.x_priv_un = None
        
        self.stats = None

    def get_stats(self):
        
        priv_dat = torch.cat([self.x_priv_tr, self.x_priv_val])
        reg_dat = torch.cat([self.x_reg_tr, self.x_reg_val])
        
        self.stats = {"Reg_min" : torch.min(reg_dat, dim = 0)[0],
                "Reg_max" : torch.max(reg_dat, dim = 0)[0] + 1e-8,
                "Priv_min" : torch.min(priv_dat, dim = 0)[0],
                "Priv_max" : torch.max(priv_dat, dim = 0)[0] + 1e-8}
                
    def create_pseudo_label(self, teacher, mode):
        if mode == "GenD":
            self.logit_teacher_tr = teacher.predict_logits(self.x_priv_tr)
            self.logit_teacher_val = teacher.predict_logits(self.x_priv_val)

            self.pseudo_logit_teacher = teacher.predict_logits(self.x_priv_un)
            self.pseudo_label_teacher = teacher.predict_label(self.x_priv_un)

        elif mode == "PFD":
            self.logit_teacher_tr = teacher.predict_logits(torch.cat([self.x_priv_tr, self.x_reg_tr], dim = -1))
            self.logit_teacher_val = teacher.predict_logits(torch.cat([self.x_priv_val, self.x_reg_val], dim = -1))

            self.pseudo_logit_teacher = teacher.predict_logits(torch.cat([self.x_priv_un, self.x_reg_un], dim = -1))
            self.pseudo_label_teacher = teacher.predict_label(torch.cat([self.x_priv_un, self.x_reg_un], dim = -1))

        else:
            print("Please choose mode from [GenD, PFD]")
            raise NotImplementedError
        self.logit_teacher_tr = self.logit_teacher_tr.float()
        self.logit_teacher_val = self.logit_teacher_val.float()
        self.pseudo_logit_teacher = self.pseudo_logit_teacher.float()

        self.pseudo_label_teacher = self.pseudo_label_teacher.long()

class labeled_dataset(Dataset):
    def __init__(self, base_dataset, _type = "train"):
        if _type == "train":
            self.x_reg = base_dataset.x_reg_tr 
            self.x_priv = base_dataset.x_priv_tr
            self.labels = base_dataset.labels_tr

        elif _type == "val":
            self.x_reg = base_dataset.x_reg_val 
            self.x_priv = base_dataset.x_priv_val
            self.labels = base_dataset.labels_val

        elif _type == "test":
            self.x_reg = base_dataset.x_reg_tst 
            self.x_priv = base_dataset.x_priv_tst
            self.labels = base_dataset.labels_tst

        else:
            raise NotImplementedError

    def __getitem__(self, index):
        
        x_reg = self.x_reg[index]
        x_priv = self.x_priv[index]
        y = self.labels[index]

        return (x_reg, x_priv, y)

    def __len__(self):
        return len(self.x_reg)



class LUPI_dataset_tr(Dataset):
    def __init__(self, base_dataset, teacher_mode=None, teacher = None, use_unlabeled = True):
        
        base_dataset.create_pseudo_label(teacher, teacher_mode)

        if use_unlabeled:
            self.x_reg = torch.cat([base_dataset.x_reg_tr, base_dataset.x_reg_un] ).float()
            self.x_priv = torch.cat([base_dataset.x_priv_tr, base_dataset.x_priv_un] ).float()
            self.labels = torch.cat([base_dataset.labels_tr, base_dataset.pseudo_label_teacher] ).long()
            self.logit_teacher = torch.cat([base_dataset.logit_teacher_tr, base_dataset.pseudo_logit_teacher] ).float()

        else:
            self.x_reg = base_dataset.x_reg_tr.float()
            self.x_priv = base_dataset.x_priv_tr.float()
            self.labels = base_dataset.labels_tr.long()
            self.logit_teacher = base_dataset.logit_teacher_tr.float()


    def __getitem__(self, index):
        
        x_reg = self.x_reg[index]
        x_priv = self.x_priv[index]

        y = (self.labels[index], self.logit_teacher[index])
        return (x_reg, x_priv, y)

    def __len__(self):
        return len(self.x_reg)
    

class LUPI_dataset_val(Dataset):
    def __init__(self, base_dataset, teacher_mode=None, teacher = None):
        
        base_dataset.create_pseudo_label(teacher, teacher_mode)

        self.x_reg = base_dataset.x_reg_val.float()
        self.x_priv = base_dataset.x_priv_val.float()
        self.labels = base_dataset.labels_val.long()
        self.logit_teacher = base_dataset.logit_teacher_val.float()


    def __getitem__(self, index):
        
        x_reg = self.x_reg[index]
        x_priv = self.x_priv[index]

        y = (self.labels[index], self.logit_teacher[index])
        return (x_reg, x_priv, y)

    def __len__(self):
        return len(self.x_reg)
    
class pretraining_dataset(Dataset):
    def __init__(self, base_dataset):
        


        self.x_reg = torch.cat([base_dataset.x_reg_tr, base_dataset.x_reg_un] ).float()
        self.x_priv = torch.cat([base_dataset.x_priv_tr, base_dataset.x_priv_un] ).float()


    def __getitem__(self, index):
        
        x_reg = self.x_reg[index]
        x_priv = self.x_priv[index]

        return (x_reg, x_priv, x_priv)


    def __len__(self):
        return len(self.x_reg)
    
def create_baseline_dataset(hpara, base_dataset):
    
    dat_tr = labeled_dataset(base_dataset, _type = "train")
    dat_val = labeled_dataset(base_dataset, _type = "val")
    dat_tst = labeled_dataset(base_dataset, _type = "test")



    D_val = DataLoader(dat_val, batch_size = hpara["batch_size"], num_workers=hpara["num_workers"])
    D_tr = DataLoader(dat_tr, batch_size = hpara["batch_size"], num_workers=hpara["num_workers"])
    D_test = DataLoader(dat_tst, batch_size = len(dat_tst), num_workers=hpara["num_workers"])
    
    return D_tr, D_val, D_test

def create_pretraining_dataset(hpara, base_dataset):

    dat_pretrain = pretraining_dataset(base_dataset)

    D_pretrain = DataLoader(dat_pretrain, batch_size=hpara["batch_size"], num_workers = hpara["num_workers"])

    return D_pretrain

def create_KD_SP(hpara, teacher, base_dataset, teacher_mode =None):

    dat_tr = LUPI_dataset_tr(base_dataset,  teacher = teacher, teacher_mode = teacher_mode, use_unlabeled = True)

    D_KD_SP = DataLoader(dat_tr, batch_size=hpara["batch_size"], num_workers=hpara["num_workers"])

    return D_KD_SP

def create_LUPI_dataset(hpara, teacher, base_dataset, teacher_mode =None):

    dat_tr = LUPI_dataset_tr(base_dataset,  teacher = teacher, teacher_mode = teacher_mode, use_unlabeled = False)
    dat_val = LUPI_dataset_val(base_dataset,  teacher = teacher, teacher_mode = teacher_mode)

    D_LUPI_tr = DataLoader(dat_tr, batch_size=hpara["batch_size"], num_workers=hpara["num_workers"])
    D_LUPI_val = DataLoader(dat_val, batch_size=hpara["batch_size"], num_workers=hpara["num_workers"])

    return D_LUPI_tr, D_LUPI_val


def create_all_datasets(hpara, base_dataset, gend_teacher, ft_data):
    # training set used for LUPI methods
    D_LUPI_tr_gend, D_LUPI_val_gend = create_LUPI_dataset(hpara, gend_teacher, base_dataset, teacher_mode ="GenD")

    D_tr_KD_SP = create_KD_SP(hpara, gend_teacher, base_dataset, teacher_mode = "GenD")


    ft_data["KD_SP_tr"] = D_tr_KD_SP
    ft_data["GenD_tr"] = D_LUPI_tr_gend
    ft_data["GenD_val"] = D_LUPI_val_gend

    return ft_data
    