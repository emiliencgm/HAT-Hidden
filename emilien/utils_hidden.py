import pandas as pd
from utils.log import create_logger
from utils.create_input_ffnn_hidden import create_input_ffnn_hidden
from utils.final_output import read_log
import subprocess
import yaml
from pathlib import Path
from utils.tantillo.final_functions import add_pred_tantillo_hidden
from utils.run_models import run_cv
from data_manager import update_dataset

data_manager_file_path = 'ours_4_FFNN.pkl'

def run_reactivity(trained_dir = 'tmp/cv_omega_data/fold_0', target_column='G_act', ensemble_size=4, input_file='input_ffnn_selectivity_hidden.pkl', save_dir='tmp/cv_omega_exp_data', features=['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']):

    path = f"reactivity_model/predict_hidden.py"
    pred_file = f"tmp/{input_file}"
    inputs = f"--pred_file {pred_file} --trained_dir {trained_dir} --save_dir {save_dir} --ensemble_size {ensemble_size} --target_column {target_column} --features {' '.join(features)}"

    with open('out_file', 'a') as out:
        subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)

    return None

def predict_hidden(test_file='tmp/species_reactivity_dataset.csv', chk_path="surrogate_model/qmdesc_wrap/model.pt"):
    #=====Make Prediction and Extract the Hidden Representations for Molecules and Radical Atoms=====
    path = "surrogate_model/predict_hidden.py"
    test_path = f"{test_file}"
    preds_path = "tmp/preds_surrogate_hidden.pkl"
    inputs = f"--test_path {test_path} --checkpoint_path {chk_path} --preds_path {preds_path}"

    with open('out_file', 'a') as out:
        subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)

#=====Define: Re-train M2 given Hidden Representations as Input=====
#NOTE 全部的tmp/input_ffnn.pkl都被用于train和valid了，没有预留test数据。test后面训练好了另加的
def run_train_M2(save_dir='reactivity_model/results/hidden', 
            data_path = None, 
            train_valid_set_path = None,
            test_set_path = None,
            trained_dir = 'reactivity_model/results/hidden/final_model_4/', 
            transfer_learning = False, 
            target_column='DG_TS_tunn',
            ensemble_size=4,
            batch_size=64,
            layers = 0,
            hidden_size=1024,
            dropout=0.0,
            lr=0.0277,
            random_state=0,
            lr_ratio=0.95,
            features=['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden'],
            max_epochs=100,
            gpu=True):
    
    path = f"reactivity_model/train_hidden.py"
    inputs = f"--save_dir {save_dir} --ensemble_size {ensemble_size} --target_column {target_column} --batch-size {batch_size}"
    if transfer_learning:
        inputs += f" --trained_dir {trained_dir} --transfer_learning"
    if data_path:
        inputs += f" --data_path {data_path}"
    if train_valid_set_path:
        inputs += f" --train_valid_set_path {train_valid_set_path}"
    if test_set_path:
        inputs += f" --test_set_path {test_set_path}"
    # inputs += f" --data_path {data_path}  --train_valid_set_path {train_valid_set_path} --test_set_path {test_set_path}"
    if gpu:
        inputs += " --gpu"
    inputs += f" --layers {layers} --hidden-size {hidden_size} --dropout {dropout} --learning_rate {lr} --lr_ratio {lr_ratio} --max-epochs {max_epochs} --random_state {random_state} --features {' '.join(features)}"

    with open('out_file', 'a') as out:
        subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)
    
    return None


def run_cv_hidden(data_path = None, 
           target_column = 'DG_TS_tunn', 
           save_dir = 'tmp/', 
           k_fold = 10,
           ensemble_size = 4, 
           sample = None, 
           transfer_learning = False,
           trained_dir = 'reactivity_model/results/final_model_4/',
           random_state = 2,
           test_set = None,
           train_valid_set = None, 
           batch_size = 64,
           layers = 0,
           hidden_size = 230,
           dropout=0.0,
           lr = 0.0277,
           max_epochs=100,
           gpu=True,
           features=['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']):
    
    path = f"reactivity_model/cross_val_hidden.py"
    inputs = f" --k_fold {k_fold} --hidden-size {hidden_size} --target_column {target_column} --batch-size {batch_size} \
    --learning_rate {lr} --lr_ratio 0.95 --random_state {random_state} --ensemble_size {ensemble_size}  --save_dir {save_dir} --layers {layers} --max-epochs {max_epochs} --dropout {dropout} --features {' '.join(features)}" 
    if gpu:
        inputs += " --gpu"
    if data_path:
         inputs += f" --data_path {data_path}"
    if sample:
         inputs += f" --sample {sample}"
    if transfer_learning:
         inputs += f" --transfer_learning --trained_dir {trained_dir}"
    if test_set and train_valid_set:
         inputs += f" --test_set_path {test_set}  --train_valid_set_path {train_valid_set}"
         

    with open('out_file', 'a') as out:
        subprocess.run(f"python {path} {inputs}", shell=True, stdout=out, stderr=out)
    
    return None


def hyper_opt_cv_hidden_in_house(layers, hidden_size, dropout, lr, logger, gpu=True, 
                        features=['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']):
    run_cv_hidden('tmp/input_ffnn_hidden.pkl', 'dG_act_corrected', 'tmp/cv_own_dataset', 10, 4, random_state=0, layers=layers, hidden_size=hidden_size, dropout=dropout, lr=lr, gpu=gpu, features=features)
    logger.info('CV done')
    logger.info('Results in tmp/cv_own_dataset/ffn_train.log')
    mae, rmse, r2 = read_log('tmp/cv_own_dataset/ffn_train.log')
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    cv = {"RMSE" : rmse, "MAE": mae, "R2": r2}
    args = {"layers":layers, "hidden_size":hidden_size, "dropout":dropout, "lr":lr}
    out_yaml = {"args": args, "10-fold cross-validation results": cv}
    out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)
    save_dir = 'tmp/cv_own_dataset'
    with open(Path(save_dir) / "cv_hyper_opt_simple.yaml", "a") as fp:
        fp.write(out_str)



def hyper_opt_cv_hidden(transfer_learning=False, trained_dir='tmp/cv_in-house_data/fold_0', features=['mol1_hidden','rad2_hidden','rad_atom2_hidden'], target_column='DFT_Barrier', save_dir='tmp/cv_tantillo_data', k_fold=4, ensemble_size=4, random_state=0, data_path = None, test_set=None, train_valid_set=None, layers=1, hidden_size_M2=1024, dropout=0.0, lr=0.0277, gpu=True, logger=None, max_epochs=100, batch_size=64, hidden_size_M1=1200, dataset_name=None, if_log=True):
    
    run_cv_hidden(transfer_learning=transfer_learning, trained_dir=trained_dir, features=features,  target_column=target_column, save_dir=save_dir, k_fold=k_fold, ensemble_size=ensemble_size, random_state=random_state, test_set=test_set, train_valid_set=train_valid_set, data_path=data_path, layers=layers, hidden_size=hidden_size_M2, dropout=dropout, lr=lr, gpu=gpu, max_epochs=max_epochs, batch_size=batch_size)

    logger.info('CV done')
    logger.info(f"Results in {save_dir}/ffn_train.log")
    mae, rmse, r2 = read_log(f"{save_dir}/ffn_train.log")
    mae = float(mae)
    rmse = float(rmse)
    r2 = float(r2)
    logger.info(f'{k_fold}-fold CV RMSE, MAE and R^2 for NN with a hidden representation: {rmse} {mae} {r2}')
    
    if if_log:
        update_dataset(file_path=data_manager_file_path, dataset_name=dataset_name, h=hidden_size_M1, rmse=rmse, mae=mae, r2=r2)

    cv = {"RMSE" : rmse, "MAE": mae, "R2": r2}
    args = {"hidden_size_M1":hidden_size_M1, "features":features, "layers":layers, "hidden_size_M2":hidden_size_M2, "dropout":dropout, "lr":lr, "transfer_learning":transfer_learning, "ensemble_size":ensemble_size, "max_epochs":max_epochs, "random_state":random_state}
    out_yaml = {"args": args, f"{k_fold}-fold cross-validation results": cv}
    out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)
    with open(Path(save_dir) / "cv_hyper_opt_simple.yaml", "a") as fp:
        fp.write(out_str)
        
    return rmse, mae, r2

def hyper_opt_up_hidden(save_dir='tmp/cv_tantillo_data', data_path = None, train_valid_set_path = 'tmp/tantillo_data/input_tantillo_hidden.pkl', test_set_path = 'tmp/tantillo_data/input_steroids_tantillo_hidden.pkl', trained_dir = 'tmp/cv_in-house_data/fold_0', transfer_learning = False, target_column='DFT_Barrier', ensemble_size=4, batch_size=64, layers = 0, hidden_size_M2=256, dropout=0.0, lr=0.0277, random_state=0, lr_ratio=0.95, features=['mol1_hidden','rad2_hidden', 'rad_atom2_hidden'], max_epochs=100, gpu=True, logger=None, hidden_size_M1=1200, dataset_name=None, if_log=True):
    
    run_train_M2(save_dir=save_dir, data_path = data_path, train_valid_set_path = train_valid_set_path, test_set_path = test_set_path, trained_dir = trained_dir, transfer_learning = transfer_learning, target_column=target_column, ensemble_size=ensemble_size, batch_size=batch_size, layers = layers, hidden_size=hidden_size_M2, dropout=dropout, lr=lr, random_state=random_state, lr_ratio=lr_ratio, features=features, max_epochs=max_epochs, gpu=gpu)

    logger.info('training done')
    logger.info(f"Results in {save_dir}/ffn_train.log")
    mae, rmse, r2 = read_log(f"{save_dir}/ffn_train.log")
    mae = float(mae)
    rmse = float(rmse)
    r2 = float(r2)
    logger.info(f'Upper bound of RMSE, MAE and R^2 for NN with a hidden representation: {rmse} {mae} {r2}')
    
    if if_log:
        update_dataset(file_path=data_manager_file_path, dataset_name=dataset_name, h=hidden_size_M1, rmse=rmse, mae=mae, r2=r2)

    metrics = {"RMSE" : rmse, "MAE": mae, "R2": r2}
    args = {"hidden_size_M1":hidden_size_M1, "features":features, "layers":layers, "hidden_size_M2":hidden_size_M2, "dropout":dropout, "lr":lr, "transfer_learning":transfer_learning, "ensemble_size":ensemble_size, "max_epochs":max_epochs, "random_state":random_state}
    out_yaml = {"args": args, f"upper bound of prediction": metrics}
    out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)
    with open(Path(save_dir) / "up_hyper_opt_simple.yaml", "a") as fp:
        fp.write(out_str)
        
    return rmse, mae, r2

def retrain_Omega_Alkoxy(args):
    # omega dataset (DOI: https://doi.org/10.1021/acsomega.2c03252)
    logger = create_logger('omega_data.log')
    logger.info('********************************')
    logger.info('======= OMEGA DATASET =======')
    # features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
    predict_hidden(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv')
    # create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/omega_data/clean_data_omega.csv', 'G_act', 'tmp/omega_data/additional_data_omega.pkl')
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/omega_data/clean_data_omega.csv', 'G_act')

    df_omega = pd.read_pickle('tmp/input_ffnn_hidden.pkl')
    df_omega = df_omega.loc[df_omega['G_act'] != 'FALSE']
    df_omega.to_pickle('tmp/input_ffnn_hidden.pkl')
    
    run_cv_hidden('tmp/input_ffnn_hidden.pkl', 'G_act', 'tmp/cv_omega_data', 10, 4, random_state=0, layers=args.layers, hidden_size=args.hidden_size, dropout=args.dropout, lr=args.lr, max_epochs=100)
    logger.info('CV done')
    logger.info('Results in tmp/cv_omega_data/ffn_train.log')
    mae, rmse, r2 = read_log('tmp/cv_omega_data/ffn_train.log')
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    cv = {"RMSE" : rmse, "MAE": mae, "R2": r2}
    args = {"layers":args.layers, "hidden_size":args.hidden_size, "dropout":args.dropout, "lr":args.lr}
    out_yaml = {"args": args, "10-fold cross-validation results": cv}
    out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)
    save_dir = 'tmp/cv_omega_data'
    with open(Path(save_dir) / "cv_hyper_opt_simple.yaml", "a") as fp:
        fp.write(out_str)


def retrain_tantillo_Cytochrome_P450(args):
    # tantillo dataset (DOI: https://doi.org/10.1002/cmtd.202100108)
    logger = create_logger('tantillo_data.log')
    logger.info('********************************')
    logger.info('======= TANTILLO DATASET =======')
    predict_hidden(test_file='tmp/tantillo_data/species_reactivity_tantillo_dataset.csv')
    logger.info('Surrogate model done')
    add_pred_tantillo_hidden(train_file='tmp/tantillo_data/clean_data_tantillo.csv',
                      test_file='tmp/tantillo_data/clean_data_steroids_tantillo.csv', 
                      pred_file='tmp/preds_surrogate_hidden.pkl')

    # NOTE　Transfer Learning loading dir: input_dim=3600 !!!
    hyper_opt_cv_hidden(transfer_learning=False, trained_dir='tmp/cv_in-house_data/fold_0', features=['mol1_hidden','rad2_hidden','rad_atom2_hidden'], target_column='DFT_Barrier', save_dir='tmp/cv_tantillo_data', k_fold=4, ensemble_size=4, random_state=0, data_path = None, test_set='tmp/tantillo_data/input_steroids_tantillo_hidden.pkl', train_valid_set='tmp/tantillo_data/input_tantillo_hidden.pkl', layers=args.layers, hidden_size=args.hidden_size, dropout=args.dropout, lr=args.lr, gpu=True, logger=logger, max_epochs=100)

    
def reproduce_RMechDB():
    logger = create_logger('rmechdb_data.log')
    logger.info('********************************')
    logger.info('======= RMechDB DATASET =======')
    
    batch_size = 64
    ensemble_size = 4
    random_state = 2
    
    run_cv(batch_size=batch_size, data_path='tmp/rmechdb_data/input_ffnn_rmechdb_pure.pkl', save_dir='tmp/cv_rmechdb_data', random_state=random_state, ensemble_size=ensemble_size, transfer_learning=True)
    mae, rmse, r2 = read_log('tmp/cv_rmechdb_data/ffn_train.log')
    logger.info(f"{ensemble_size} ensemble(s), batch_size: {batch_size}, random_state: {random_state}")
    logger.info(f'<<pure>> 10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    
    run_cv(batch_size=batch_size, data_path='tmp/rmechdb_data/input_ffnn_rmechdb_water.pkl', save_dir='tmp/cv_rmechdb_data', random_state=random_state, ensemble_size=ensemble_size, transfer_learning=True)
    mae, rmse, r2 = read_log('tmp/cv_rmechdb_data/ffn_train.log')
    logger.info(f"{ensemble_size} ensemble(s), batch_size: {batch_size}, random_state: {random_state}")
    logger.info(f'<<water>> 10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    
    run_cv(batch_size=batch_size, data_path='tmp/rmechdb_data/input_ffnn_rmechdb_water_peroxide.pkl', save_dir='tmp/cv_rmechdb_data', random_state=random_state, ensemble_size=ensemble_size, transfer_learning=True)
    mae, rmse, r2 = read_log('tmp/cv_rmechdb_data/ffn_train.log')
    logger.info(f"{ensemble_size} ensemble(s), batch_size: {batch_size}, random_state: {random_state}")
    logger.info(f'<<water_peroxide>> 10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

