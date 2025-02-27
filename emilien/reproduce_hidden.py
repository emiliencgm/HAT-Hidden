import pandas as pd
from utils.log import create_logger_emilien
from utils.input_for_pred import create_input_pred
from utils.create_input_ffnn_hidden import create_input_ffnn_hidden
from utils.final_output import read_log, get_stats
import yaml
from pathlib import Path
from utils.tantillo.final_functions import add_pred_tantillo_hidden
from emilien.utils_hidden import predict_hidden, hyper_opt_cv_hidden, run_cv_hidden, hyper_opt_up_hidden, run_reactivity

from data_manager import update_dataset

save_dir_in_house = 'tmp/cv_in-house_data'
trained_dir_in_house = 'tmp/cv_in-house_data/fold_0'

save_dir_omega = 'tmp/cv_omega_60test'
trained_dir_omega = 'tmp/cv_omega_60test/fold_0'

save_dir_tantillo = 'tmp/cv_tantillo_data'

save_dir_hong = 'tmp/cv_hong_data'
trained_dir_hong = 'tmp/cv_hong_data/fold_0'

save_dir_exp_omega_selectivity = 'tmp/omega_exp_data'

save_dir_bietti = 'tmp/cv_bietti_data'

save_dir_rmechdb = 'tmp/cv_rmechdb_data'

data_manager_file_path = 'ours_4_FFNN.pkl'

def retrain_in_house(args, if_log=True):
    # # cross-validation in-house HAT dataset
    if if_log:
        logger = create_logger_emilien('in-house_data.log')
    else:
        logger = create_logger_emilien('trash_in-house.log')
    logger.info('********************************')
    logger.info(f'======= In-HOUSE DATASET, M1\'s hidden size: {args.chk_path_hidden}=======')
    df = pd.read_csv('tmp/own_dataset/reactivity_database_corrected.csv', index_col=0)
    create_input_pred(df=df, target_column='dG_act_corrected', target_column_2='dG_rxn')
    predict_hidden(chk_path=args.chk_path)
    logger.info('Surrogate model done')
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/reactivity_database_mapped.csv', 'dG_act_corrected')

    # hyper_opt_cv_hidden_in_house(args.layers, args.hidden_size, args.dropout, args.lr, logger, features=args.features)
    rmse, mae, r2 = hyper_opt_cv_hidden(features=args.features, target_column='dG_act_corrected', save_dir=save_dir_in_house, k_fold=10, ensemble_size=4, data_path='tmp/input_ffnn_hidden.pkl', layers=args.layers, hidden_size_M2=args.hidden_size, dropout=args.dropout, lr=args.lr, logger=logger, random_state=args.random_state, hidden_size_M1=args.chk_path_hidden, dataset_name='in-house')
    
    return rmse, mae, r2

def retrain_Omega_Alkoxy_reprod(args, if_log=True):
    # omega dataset (DOI: https://doi.org/10.1021/acsomega.2c03252)
    if if_log:
        logger = create_logger_emilien('omega_data.log')
    else:
        logger = create_logger_emilien('trash_omega.log')
    logger.info('********************************')
    logger.info(f'======= OMEGA DATASET, M1\'s hidden size: {args.chk_path_hidden} =======')
    predict_hidden(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv', chk_path=args.chk_path)
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/omega_data/clean_data_omega.csv', 'G_act')

    df_omega = pd.read_pickle('tmp/input_ffnn_hidden.pkl')
    df_train = df_omega.iloc[:240]
    df_train = df_train.loc[df_train['G_act'] != 'FALSE']
    df_test = df_omega.iloc[240:]
    df_test = df_test.loc[df_test['G_act'] != 'FALSE']
    df_train.to_pickle('tmp/input_ffnn_hidden.pkl')
    df_train.to_csv('tmp/omega_data/train_valid_set.csv')
    df_test.to_csv('tmp/omega_data/test_set.csv')
    df_train.to_pickle('tmp/omega_data/train_valid_set.pkl')
    df_test.to_pickle('tmp/omega_data/test_set.pkl')
    
    rmse, mae, r2 = hyper_opt_cv_hidden(transfer_learning=args.transfer_learning, features=args.features, target_column='G_act', save_dir=save_dir_omega, k_fold=10, ensemble_size=4, random_state=args.random_state, test_set='tmp/omega_data/test_set.pkl', train_valid_set='tmp/omega_data/train_valid_set.pkl', layers=args.layers, hidden_size_M2=args.hidden_size, dropout=args.dropout, lr=args.lr, gpu=True, logger=logger, max_epochs=100, batch_size=64, hidden_size_M1=args.chk_path_hidden, dataset_name='omega')

    return rmse, mae, r2
    
    # run_cv_hidden(target_column='G_act', save_dir='tmp/cv_omega_60test', k_fold=10, ensemble_size=4, transfer_learning=args.transfer_learning, test_set='tmp/omega_data/test_set.pkl', train_valid_set='tmp/omega_data/train_valid_set.pkl', random_state=args.random_state, batch_size=64, layers=args.layers, hidden_size=args.hidden_size, dropout=args.dropout, lr=args.lr, max_epochs=100, features=args.features)
    # logger.info('CV done')
    # logger.info('Results in tmp/cv_omega_60test/ffn_train.log')
    # mae, rmse, r2 = read_log('tmp/cv_omega_60test/ffn_train.log')
    # logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation (60 test): {rmse} {mae} {r2}')

    # cv = {"RMSE" : rmse, "MAE": mae, "R2": r2}
    # args = {"layers":args.layers, "hidden_size":args.hidden_size, "dropout":args.dropout, "lr":args.lr}
    # out_yaml = {"args": args, "10-fold cross-validation results": cv}
    # out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)
    # save_dir = 'tmp/cv_omega_60test'
    # with open(Path(save_dir) / "cv_hyper_opt_simple.yaml", "a") as fp:
    #     fp.write(out_str)

def retrain_tantillo_Cytochrome_P450_reprod(args, if_log=True):
    # tantillo dataset (DOI: https://doi.org/10.1002/cmtd.202100108)
    if if_log:
        logger = create_logger_emilien('tantillo_data.log')
    else:
        logger = create_logger_emilien('trash_tantillo.log')
    logger.info('********************************')
    logger.info(f'======= TANTILLO DATASET, M1\'s hidden size: {args.chk_path_hidden} =======')
    predict_hidden(test_file='tmp/tantillo_data/species_reactivity_tantillo_dataset.csv', chk_path=args.chk_path)
    logger.info('Surrogate model done')
    add_pred_tantillo_hidden(train_file='tmp/tantillo_data/clean_data_tantillo.csv',
                      test_file='tmp/tantillo_data/clean_data_steroids_tantillo.csv', 
                      pred_file='tmp/preds_surrogate_hidden.pkl')
    
    # NOTE　Transfer Learning loading dir: input_dim=3600 !!!
    #@@@ train : test = 18 : 6
    # hyper_opt_up_hidden(save_dir='tmp/cv_tantillo_data', data_path = None, train_valid_set_path = 'tmp/tantillo_data/input_tantillo_hidden.pkl', test_set_path = 'tmp/tantillo_data/input_steroids_tantillo_hidden.pkl', trained_dir = 'tmp/cv_in-house_data/fold_0', transfer_learning = False, target_column='DFT_Barrier', ensemble_size=4, batch_size=64, layers = args.layers, hidden_size=args.hidden_size, dropout=args.dropout, lr=args.lr, random_state=0, lr_ratio=0.95, features=['mol1_hidden','rad2_hidden', 'rad_atom2_hidden'], max_epochs=100, gpu=True, logger=logger)
    
    #@@@ train : val : test = 16 : 2 : 6
    rmse, mae, r2 = hyper_opt_up_hidden(save_dir=save_dir_tantillo, data_path = 'tmp/tantillo_data/input_tantillo_hidden.pkl', train_valid_set_path = None, test_set_path = 'tmp/tantillo_data/input_steroids_tantillo_hidden.pkl', trained_dir = trained_dir_in_house, transfer_learning = args.transfer_learning, target_column='DFT_Barrier', ensemble_size=4, batch_size=64, layers = args.layers, hidden_size_M2=args.hidden_size, dropout=args.dropout, lr=args.lr, random_state=args.random_state, lr_ratio=0.95, features=args.features, max_epochs=100, gpu=True, logger=logger, hidden_size_M1=args.chk_path_hidden, dataset_name='tantillo')

    return rmse, mae, r2

def retrain_Hong_Photoredox(args, if_log=True):
    if if_log:
        logger = create_logger_emilien('hong_data.log')
    else:
        logger = create_logger_emilien('trash_hong.log')
    logger.info('********************************')
    logger.info(f'======= HONG DATASET, M1\'s hidden size: {args.chk_path_hidden} =======')
    df_hong = pd.read_csv('tmp/hong_data/training_hong_clean.csv', index_col=0)
    create_input_pred(df_hong, 'DG_TS')
    predict_hidden(chk_path=args.chk_path)
    logger.info('Surrogate model done')
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS')

    # run_cv_hidden('tmp/input_ffnn_hidden.pkl', 'DG_TS', 'tmp/cv_hong_data', 5, 4, batch_size=64, layers=args.layers, hidden_size=args.hidden_size, dropout=args.dropout, lr=args.lr)
    # logger.info('CV done with 4 ensembles and no transfer learning')
    # logger.info('Results in tmp/cv_hong_data/ffn_train.log')
    # mae, rmse, r2 = read_log('tmp/cv_hong_data/ffn_train.log')
    # logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB hidden representation, all datapoints, 4 ensembles and no transfer learning (5-fold!!!): {rmse} {mae} {r2}')
    rmse, mae, r2 = hyper_opt_cv_hidden(data_path='tmp/input_ffnn_hidden.pkl', target_column='DG_TS', save_dir=save_dir_hong, k_fold=5, ensemble_size=4, layers=args.layers, hidden_size_M2=args.hidden_size, dropout=args.dropout, lr=args.lr, features=args.features, logger=logger, random_state=args.random_state, hidden_size_M1=args.chk_path_hidden, dataset_name='hong')
    
    return rmse, mae, r2

def retrain_exp_Omega_alkoxy(args, if_log=True):
    if if_log:
        logger = create_logger_emilien('omega_exp_data.log')
    else:
        logger = create_logger_emilien('trash_selectivity.log')
    logger.info('********************************')
    logger.info(f'======= OMEGA_EXP DATASET, M1\'s hidden size: {args.chk_path_hidden} =======')
    predict_hidden(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv', chk_path=args.chk_path)
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/omega_data/clean_data_omega_selectivity.csv', 'gibbs', output='input_ffnn_selectivity')
    
    df_selectivity = pd.read_pickle('tmp/input_ffnn_selectivity_hidden.pkl')
    df_selectivity.rename(columns={'gibbs': 'G_act'}, inplace=True)
    df_selectivity.to_pickle('tmp/input_ffnn_selectivity_hidden.pkl')
    
    run_reactivity(trained_dir=trained_dir_omega, target_column='G_act', ensemble_size=4, input_file='input_ffnn_selectivity_hidden.pkl', save_dir=save_dir_exp_omega_selectivity, features=args.features)
    
    df_selectivity_pred = pd.read_csv(save_dir_exp_omega_selectivity+'/pred.csv')
    mae, rmse, r2 = get_stats(df_selectivity['G_act'], df_selectivity_pred['DG_TS_tunn'])
    
    logger.info(f"args: {args}")
    logger.info(f'RMSE, MAE and R^2 for NN (**4** ensembles trained on Omega and predict on Omega_Exp) with a hidden representation: {rmse} {mae} {r2}')
    
    update_dataset(file_path=data_manager_file_path, dataset_name='selectivity', h=args.chk_path_hidden, rmse=rmse, mae=mae, r2=r2)
    
    return rmse, mae, r2
    
    
def retrain_omega_bietti_hong(args, if_log=True):
    '''
    Bietti's dataset is "outliers" of Omega, while Javier pre-trained M2 on Hong's to predict on Bietti's. 
    '''
    if if_log:
        logger = create_logger_emilien('omega_bietti_hong_data.log')
    else:
        logger = create_logger_emilien('trash_bietti.log')
    logger.info('********************************')
    logger.info(f'======= OMEGA_BIETTI_HONG DATASET, M1\'s hidden size: {args.chk_path_hidden} =======')
    predict_hidden(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv', chk_path=args.chk_path)
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/omega_data/clean_data_omega_exp.csv', 'gibbs_exp')
    # NOTE Seems to be negative transfer
    rmse, mae, r2 = hyper_opt_cv_hidden(transfer_learning=args.transfer_learning, trained_dir=trained_dir_hong, features=args.features, target_column='gibbs_exp', save_dir=save_dir_bietti, k_fold=10, ensemble_size=4, data_path='tmp/input_ffnn_hidden.pkl', layers=args.layers, hidden_size_M2=args.hidden_size, dropout=args.dropout, lr=args.lr, logger=logger, random_state=args.random_state, hidden_size_M1=args.chk_path_hidden, dataset_name='bietti')
    
    return rmse, mae, r2
    
def retrain_RMechDB(args, if_log=True):
    if if_log:
        logger = create_logger_emilien('rmechdb_data.log')
    else:
        logger = create_logger_emilien('trash_rmechdb.log')
    logger.info('********************************')
    logger.info(f'======= RMechDB DATASET, M1\'s hidden size: {args.chk_path_hidden} =======')
    df_rmechdb = pd.read_pickle('tmp/rmechdb_data/clean_data_RMechDB.pkl')
    create_input_pred(df_rmechdb, 'DG_TS_tunn')
    predict_hidden(chk_path=args.chk_path)
    logger.info('Surrogate model done')
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS_tunn')
    rmse, mae, r2 = hyper_opt_cv_hidden(transfer_learning=args.transfer_learning, trained_dir=trained_dir_in_house, data_path='tmp/input_ffnn_hidden.pkl', target_column='DG_TS_tunn', save_dir=save_dir_rmechdb, k_fold=10, ensemble_size=4, layers=args.layers, hidden_size_M2=args.hidden_size, dropout=args.dropout, lr=args.lr, features=args.features, logger=logger, random_state=args.random_state, hidden_size_M1=args.chk_path_hidden, dataset_name='rmechdb')  
    
    return rmse, mae, r2  
    





