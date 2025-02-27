from emilien.reproduce_hidden import retrain_in_house, retrain_Omega_Alkoxy_reprod, retrain_exp_Omega_alkoxy, retrain_Hong_Photoredox, retrain_omega_bietti_hong, retrain_tantillo_Cytochrome_P450_reprod, retrain_RMechDB
from emilien.utils_hidden import reproduce_RMechDB
import argparse
import shutil
import os

def remove_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder {folder_path} is removed.")
    else:
        print(f"{folder_path} not exist.")
        
def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} is removed")
    else:
        print(f"{file_path} not exist.")



def reproduce_hidden(dataset='in_house', args=None, fine_tune=False):
    '''
    dataset: in_house, omega, omega_exp, hong, hong_bietti, tantillo, atmospheric \n
    fine_tune=True: args will be changed;\n
    fine_tune=False: args not changed regardless of input args.\n
    '''
    
    if dataset == 'in_house':
        if not fine_tune:
            args.hidden_size = 1024
            args.layers = 1
            args.dropout = 0.0
            args.lr = 0.0277
            args.features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
            args.transfer_learning = False
            args.random_state = 0
        # Simple 10-fold CV
        rmse, mae, r2 = retrain_in_house(args)
            
    if dataset == 'omega':
        if not fine_tune:
            args.hidden_size = 1024
            args.layers = 0
            args.dropout = 0.0
            args.lr = 0.0277
            args.features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
            args.transfer_learning = False
            args.random_state = 0
        # Pre-selected 60 for test and 238 for train-validation, "10-fold" to change validation set.
        rmse, mae, r2 = retrain_Omega_Alkoxy_reprod(args)
            
    if dataset == 'omega_exp':
        if not fine_tune:
            args.hidden_size = 1024
            args.layers = 0
            args.dropout = 0.0
            args.lr = 0.0277
            args.features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
            args.transfer_learning = True
            args.random_state = 0
        # directly test on Exp.Omega using the M2 trained on Omega.
        
        for i in range(10):
            folder_path = f"tmp/cv_omega_60test/fold_{i}"
            remove_folder(folder_path)
            file_path = f"tmp/cv_omega_60test/test_predicted_{i}.csv"
            remove_file(file_path)
        
        args.transfer_learning = False
        retrain_Omega_Alkoxy_reprod(args, if_log=False)
        args.transfer_learning = True
        rmse, mae, r2 = retrain_exp_Omega_alkoxy(args)
        
    
    if dataset == 'hong':
        if not fine_tune:
            args.hidden_size = 1024
            args.layers = 1
            args.dropout = 0.0
            args.lr = 0.0277
            args.features = ['rad_atom1_hidden','rad_atom2_hidden']
            args.transfer_learning = False
            args.random_state = 0
        # simple 5-fold CV
        rmse, mae, r2 = retrain_Hong_Photoredox(args)
            
    if dataset == 'omega_bietti_hong':
        if not fine_tune:
            args.hidden_size = 1024
            args.layers = 1
            args.dropout = 0.0
            args.lr = 0.0277
            args.features = ['rad_atom1_hidden','rad_atom2_hidden'] # TODO can change
            args.transfer_learning = False # TODO seems to be negative transfer
            args.random_state = 0
        # pre-train on Hong and then 10-fold CV on bietti
        # retrain_Hong_Photoredox(args)
        rmse, mae, r2 = retrain_omega_bietti_hong(args)
            
    if dataset == 'tantillo':
        if not fine_tune:
            args.hidden_size = 256
            args.layers = 0
            args.dropout = 0.0
            args.lr = 0.0277
            args.features = ['mol1_hidden', 'rad2_hidden', 'rad_atom2_hidden']
            args.transfer_learning = False
            args.random_state = 0
        # pre-selected 6 for test, randomly select 2 for validation and the left 16 for training from scratch or pre-trained on In-House.
        # retrain_in_house(args)
        rmse, mae, r2 = retrain_tantillo_Cytochrome_P450_reprod(args)

    if dataset == 'rmechdb':
        if not fine_tune:
            args.hidden_size = 230
            args.layers = 0
            args.dropout = 0.
            args.lr = 0.0277
            args.features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
            args.transfer_learning = True
            args.random_state = 0
        # simple 10-fold CV on RMechDB (training from scratch or pre-training on In-House)
        for i in range(10):
            folder_path = f"tmp/cv_in-house_data/fold_{i}"
            remove_folder(folder_path)
            file_path = f"tmp/cv_in-house_data/test_predicted_{i}.csv"
            remove_file(file_path)
        
        args.transfer_learning = False
        retrain_in_house(args, if_log=False)
        args.transfer_learning = True
        rmse, mae, r2 = retrain_RMechDB(args)
        
    return rmse, mae, r2


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='in_house', type=str)
    parser.add_argument("--hidden_size", default=1024, type=int, help='hidden size of M2')
    parser.add_argument("--layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=0.0277, type=float)
    parser.add_argument("--features", nargs="+", type=str,  default=['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden'])
    parser.add_argument("--random_state", default=0, type=int)
    parser.add_argument("--transfer_learning", default=0, type=int)
    parser.add_argument("--chk_path", default="surrogate_model/qmdesc_wrap/model.pt", type=str)
    parser.add_argument("--chk_path_hidden", default=1200, type=int, help='hidden size of M1')
    parser.add_argument("--fine_tune", default=0, type=int)
    args = parser.parse_args()
    
    if args.transfer_learning == 0:
        args.transfer_learning = False
    elif args.transfer_learning == 1:
        args.transfer_learning = True
    else:
        raise Exception("transfer_learning argument error")
    
    # TODO :NOTE
    args.chk_path = f"surrogate_model/output_h{args.chk_path_hidden}_b50_e100/model_0/model.pt"
    
    rmse, mae, r2 = reproduce_hidden(dataset=args.dataset, args=args, fine_tune=args.fine_tune)
    
    




