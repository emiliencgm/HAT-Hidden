import pandas as pd
from utils.log import create_logger_emilien
from utils.input_for_pred import create_input_pred
from utils.run_models import run_surrogate, run_cv, run_train, run_reactivity
from utils.create_input_ffnn import create_input_ffnn
from utils.final_output import read_log, get_stats
from utils.baseline.final_functions import get_cross_val_accuracy_ada_boost_regression, get_cross_val_accuracy_rf_descriptors
from utils.baseline.final_functions import get_accuracy_linear_regression, get_accuracy_rf_descriptors
from utils.tantillo.final_functions import add_pred_tantillo
from utils.omega.final_functions import exp_data_corr
from utils.common import prepare_df
from emilien.utils_hidden import reproduce_RMechDB
import argparse

from data_manager import update_dataset
# NOTE 用于验证猜想：改变M1的hidden_size时，根据desc预测和根据hidden预测的M2的性能变化趋势会有不同——hidden space可能更加鲁棒，因此不需要特别准确的desc预测结果也可以预测activation energy。

def reproduce(hidden_size_M1=1200):
    
    data_manager_file_path = 'reproduce_4_FFNN.pkl'
    
    checkpoint_path = f"surrogate_model/output_h{hidden_size_M1}_b50_e100/model_0/model.pt"
    
    # cross-validation in-house HAT dataset
    logger = create_logger_emilien(f'reproduce_emilien.log')
    
    logger_simple = create_logger_emilien(f'reproduce_emilien_simple.log')
    logger_simple.info('='*30)
    logger_simple.info('Only the results that are expected to be of interset are recorded here.')
    logger_simple.info(f'hidden size of M1: {hidden_size_M1}')
    
    logger.info('********************************')
    logger.info(f'======= In-HOUSE DATASET M1 hidden_size={hidden_size_M1} =======')
    df = pd.read_csv('tmp/own_dataset/reactivity_database_corrected.csv', index_col=0)
    create_input_pred(df=df, target_column='dG_act_corrected', target_column_2='dG_rxn')
    run_surrogate(chk_path=checkpoint_path)
    logger.info('Surrogate model done')
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'dG_act_corrected')

    run_cv('tmp/input_ffnn.pkl', 'dG_act_corrected', 'tmp/cv_own_dataset', 10, 1, random_state=0)
    logger.info('CV done')
    logger.info('Results in tmp/cv_own_dataset/ffn_train.log')
    mae, rmse, r2 = read_log('tmp/cv_own_dataset/ffn_train.log')
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    run_cv('tmp/input_ffnn.pkl', 'dG_act_corrected', 'tmp/cv_own_dataset_4', 10, 4, random_state=0)
    logger.info('CV done')
    logger.info('Results in tmp/cv_own_dataset_4/ffn_train.log')
    mae, rmse, r2 = read_log('tmp/cv_own_dataset_4/ffn_train.log')
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation and 4 ensembles: {rmse} {mae} {r2}')
    logger_simple.info('**********IN-HOUSE**********')
    logger_simple.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation and **4** ensembles: {rmse} {mae} {r2}')
    update_dataset(file_path=data_manager_file_path, dataset_name='in-house', h=hidden_size_M1, rmse=rmse, mae=mae, r2=r2)

    # tantillo dataset (DOI: https://doi.org/10.1002/cmtd.202100108)
    # logger = create_logger('tantillo_data.log')
    logger.info('********************************')
    logger.info(f'======= TANTILLO DATASET  M1 hidden_size={hidden_size_M1}=======')
    features = ['s_rad', 'Buried_Vol']
    run_surrogate(test_file='tmp/tantillo_data/species_reactivity_tantillo_dataset.csv', chk_path=checkpoint_path)
    logger.info('Surrogate model done')
    add_pred_tantillo(train_file='tmp/tantillo_data/clean_data_tantillo.csv',
                      test_file='tmp/tantillo_data/clean_data_steroids_tantillo.csv', 
                      pred_file='tmp/preds_surrogate.pkl')
    df_train_tantillo = pd.read_pickle('tmp/tantillo_data/input_tantillo.pkl')
    df_test_tantillo = pd.read_pickle('tmp/tantillo_data/input_steroids_tantillo.pkl')
    df_train_tantillo = prepare_df(df_train_tantillo, features)
    df_test_tantillo = prepare_df(df_test_tantillo, features)
    get_accuracy_linear_regression(df_train_tantillo, df_test_tantillo, logger, 'DFT_Barrier')
    logger_simple.info('**********TANTILLO**********')
    get_accuracy_linear_regression(df_train_tantillo, df_test_tantillo, logger_simple, 'DFT_Barrier')
    
    #========== TODO FFNN on tantillo implemented by Emilien. ==========
    # NOTE change inputs "--hidden-size 200 --features Buried_Vol s_rad" in run_train()
    # change inputs "--features Buried_Vol s_rad" in run_reactivity()
    # change df = pd.read_pickle(args.pred_file) of reactivity_model.predict.py
    
    # df_train_tantillo.to_pickle('tmp/tantillo_data/input_tantillo_train_valid.pkl')
    # df_test_tantillo.to_pickle('tmp/tantillo_data/input_steroids_tantillo_test.pkl')
    
    # run_train(save_dir='tmp/reprd_tantillo_data_emilien', data_path='tmp/tantillo_data/input_tantillo_train_valid.pkl', trained_dir='reactivity_model/results/final_model_4/', transfer_learning=False, target_column='DFT_Barrier', ensemble_size=4, batch_size=64)
    
    # run_reactivity(trained_dir='tmp/reprd_tantillo_data_emilien', target_column='DFT_Barrier', ensemble_size=4, input_file='tantillo_data/input_steroids_tantillo_test.pkl')
    
    # df_tantillo_pred = pd.read_csv('tmp/pred.csv')
    # mae, rmse, r2 = get_stats(df_test_tantillo['DFT_Barrier'], df_tantillo_pred['DG_TS_tunn'])
    # logger.info("Tantillo 4 ensemble of 200-neuron FFNN from scratch")
    # logger.info(f'RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    # logger_simple.info('**********TANTILLO**********')
    # logger_simple.info(f'RMSE, MAE and R^2 for NN with a learned-VB representation and 4 ensembles: {rmse} {mae} {r2}')
    # update_dataset(file_path=data_manager_file_path, dataset_name='tantillo', h=hidden_size_M1, rmse=rmse, mae=mae, r2=r2)
    
    
    # omega dataset (DOI: https://doi.org/10.1021/acsomega.2c03252)
    # logger = create_logger('omega_data.log')
    logger.info('********************************')
    logger.info(f'======= OMEGA DATASET M1 hidden_size={hidden_size_M1} =======')
    features = ['dG_forward', 'dG_reverse', 'q_reac0', 'qH_reac0', 'q_reac1', 's_reac1', 'q_prod0', 's_prod0', 'q_prod1', 'qH_prod1', 'BV_reac1', 'BV_prod0', 'fr_dG_forward', 'fr_dG_reverse']
    run_surrogate(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv', chk_path=checkpoint_path)
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/omega_data/clean_data_omega.csv', 'G_act', 'tmp/omega_data/additional_data_omega.pkl')
    df_omega = pd.read_pickle('tmp/input_ffnn.pkl')
    df_train = df_omega.iloc[:240]
    df_train = df_train.loc[df_train['G_act'] != 'FALSE']
    df_test = df_omega.iloc[240:]
    df_train.to_pickle('tmp/input_ffnn.pkl')
    df_train = prepare_df(df_train, features)
    df_test = prepare_df(df_test, features)
    logger.info('Train set')
    get_accuracy_linear_regression(df_train, df_train, logger, 'G_act')
    logger.info('Test set')
    get_accuracy_linear_regression(df_train, df_test, logger, 'G_act')
    parameters_rf = {'n_estimators': 300, 'max_features': 1, 'min_samples_leaf': 1}
    get_accuracy_rf_descriptors(df_train, df_test, logger, parameters_rf, 'G_act')
    
    logger_simple.info('**********OMEGA***********')
    get_accuracy_rf_descriptors(df_train, df_test, logger_simple, parameters_rf, 'G_act')

    df_train.to_csv('tmp/omega_data/train_valid_set.csv')
    df_test.to_csv('tmp/omega_data/test_set.csv')
    df_test.to_pickle('tmp/omega_data/test_set.pkl')
    run_cv(target_column='G_act', save_dir='tmp/cv_omega_TF_4', k_fold=10, ensemble_size=4, transfer_learning=True, test_set='tmp/omega_data/test_set.pkl', train_valid_set='tmp/omega_data/train_valid_set.csv', random_state=0, batch_size=24)
    mae, rmse, r2 = read_log('tmp/cv_omega_TF_4/ffn_train.log')
    logger.info("4 ensembles and transfer learning")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    logger_simple.info(f'**4** ensembles and transfer learning on in-house, 10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    update_dataset(file_path=data_manager_file_path, dataset_name='omega', h=hidden_size_M1, rmse=rmse, mae=mae, r2=r2)
    
    run_cv(target_column='G_act', save_dir='tmp/cv_omega_no_TF_4', k_fold=10, ensemble_size=4, test_set='tmp/omega_data/test_set.pkl', train_valid_set='tmp/omega_data/train_valid_set.csv', random_state=0, batch_size=24)
    mae, rmse, r2 = read_log('tmp/cv_omega_no_TF_4/ffn_train.log')
    logger.info("4 ensembles")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    run_cv(target_column='G_act', save_dir='tmp/cv_omega_no_TF_1', k_fold=10, ensemble_size=1, test_set='tmp/omega_data/test_set.pkl', train_valid_set='tmp/omega_data/train_valid_set.csv', random_state=0, batch_size=24)
    mae, rmse, r2 = read_log('tmp/cv_omega_no_TF_1/ffn_train.log')
    logger.info("1 ensemble")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    run_cv(target_column='G_act', save_dir='tmp/cv_omega_TF_1', k_fold=10, ensemble_size=1, transfer_learning=True, test_set='tmp/omega_data/test_set.pkl', train_valid_set='tmp/omega_data/train_valid_set.csv', random_state=0, trained_dir='reactivity_model/results/final_model_1/', batch_size=24)
    mae, rmse, r2 = read_log('tmp/cv_omega_TF_1/ffn_train.log')
    logger.info("1 ensemble and transfer learning")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    logger_simple.info(f'**1** ensemble and transfer learning on in-house, 10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    # Exp. Omega  selectivity.=========
    run_train(save_dir='tmp/final_model_omega_4', data_path='tmp/input_ffnn.pkl', target_column='G_act', batch_size=24)
    run_train(save_dir='tmp/final_model_omega_1', data_path='tmp/input_ffnn.pkl', target_column='G_act', batch_size=24, ensemble_size=1, trained_dir='reactivity_model/results/final_model_1/')

    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/omega_data/clean_data_omega_selectivity.csv', 'gibbs', 'tmp/omega_data/additional_data_omega.pkl', 'input_ffnn_selectivity')
    df_selectivity = pd.read_pickle('tmp/input_ffnn_selectivity.pkl')
    df_selectivity.rename(columns={'gibbs': 'G_act'}, inplace=True)
    df_selectivity = prepare_df(df_selectivity, features)
    get_accuracy_linear_regression(df_train, df_selectivity, logger, 'G_act', print_pred=True, name_out='pred_selectivity_lm')
    get_accuracy_rf_descriptors(df_train, df_selectivity, logger, parameters_rf, 'G_act', print_pred=True, name_out='pred_selectivity_rf')
    logger_simple.info('**********EXP OMEGA SELECTIVITY**********')
    get_accuracy_rf_descriptors(df_train, df_selectivity, logger_simple, parameters_rf, 'G_act', print_pred=True, name_out='pred_selectivity_rf')
    df_selectivity.to_pickle('tmp/input_ffnn_selectivity.pkl')
    df_selectivity.to_csv('tmp/input_ffnn_selectivity.csv')
    run_reactivity(trained_dir='tmp/final_model_omega_4/', target_column='G_act', ensemble_size=4, input_file='input_ffnn_selectivity.csv')
    df_selectivity_pred = pd.read_csv('tmp/pred.csv')
    mae, rmse, r2 = get_stats(df_selectivity['G_act'], df_selectivity_pred['DG_TS_tunn'])
    logger.info("Selectivity 4 ensemble")
    logger.info(f'RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    logger_simple.info(f'**4** ensembles and pre-trained on OMEGA, direct test RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    update_dataset(file_path=data_manager_file_path, dataset_name='selectivity', h=hidden_size_M1, rmse=rmse, mae=mae, r2=r2)

    run_reactivity(trained_dir='tmp/final_model_omega_1/', target_column='G_act', ensemble_size=1, input_file='input_ffnn_selectivity.csv')
    df_selectivity_pred = pd.read_csv('tmp/pred.csv')
    mae, rmse, r2 = get_stats(df_selectivity['G_act'], df_selectivity_pred['DG_TS_tunn'])
    logger.info("Selectivity 1 ensemble")
    logger.info(f'RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    logger_simple.info(f'**1** ensemble and pre-trained on OMEGA, direct test RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/omega_data/clean_data_omega_exp.csv', 'gibbs_exp', 'tmp/omega_data/additional_data_omega.pkl', output='input_ffnn_bietti')
    
    # cross-validation Hong data (DOI https://doi.org/10.1039/D1QO01325D) Photoredox HAT ^60
    # logger = create_logger('hong_data.log')
    logger.info('********************************')
    logger.info(f'======= HONG DATASET M1 hidden_size={hidden_size_M1} =======')
    df_hong = pd.read_csv('tmp/hong_data/training_hong_clean.csv', index_col=0)
    create_input_pred(df_hong, 'DG_TS')
    run_surrogate(chk_path=checkpoint_path)
    logger.info('Surrogate model done')
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS')
    
    for sample in [25, 50, 100, 200, 300, 400]:
        save_dir = f"tmp/cv_hong_{sample}"
        run_cv(data_path='tmp/input_ffnn.pkl', target_column='DG_TS', save_dir=save_dir, k_fold=5, ensemble_size=1, sample=sample, batch_size=32)
        logger.info(f'CV done with {sample} datapoints')
        logger.info(f'Results in {save_dir}/ffn_train.log')
        mae, rmse, r2 = read_log(f'{save_dir}/ffn_train.log')
        logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation and {sample} datapoints: {rmse} {mae} {r2}')

        save_dir = f"tmp/cv_hong_{sample}_4_TF"
        run_cv(data_path='tmp/input_ffnn.pkl', target_column='DG_TS', save_dir=save_dir, k_fold=5, ensemble_size=4, sample=sample, transfer_learning=True, batch_size=32)
        logger.info(f'CV done with {sample} datapoints, 4 ensembles and transfer learning')
        logger.info(f'Results in {save_dir}/ffn_train.log')
        mae, rmse, r2 = read_log(f'{save_dir}/ffn_train.log')
        logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation, {sample} datapoints, 4 ensembles and transfer learning: {rmse} {mae} {r2}')

    run_cv('tmp/input_ffnn.pkl', 'DG_TS', 'tmp/cv_hong', 5, 1, batch_size=32)
    logger.info('CV done')
    logger.info('Results in tmp/cv_hong/ffn_train.log')
    mae, rmse, r2 = read_log('tmp/cv_hong/ffn_train.log')
    logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation and all datapoints: {rmse} {mae} {r2}')

    run_cv('tmp/input_ffnn.pkl', 'DG_TS', 'tmp/cv_hong_4', 5, 4, batch_size=32)
    logger.info('CV done with 4 ensembles and transfer learning')
    logger.info('Results in tmp/cv_hong_4/ffn_train.log')
    mae, rmse, r2 = read_log('tmp/cv_hong_4/ffn_train.log')
    logger.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation, all datapoints, 4 ensembles and transfer learning:: {rmse} {mae} {r2}')
    logger_simple.info('**********HONG**********')
    logger_simple.info(f'5-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation, all datapoints, 4 ensembles and transfer learning on in-house: {rmse} {mae} {r2}')
    update_dataset(file_path=data_manager_file_path, dataset_name='hong', h=hidden_size_M1, rmse=rmse, mae=mae, r2=r2)

    df_hong_desc = pd.read_pickle('tmp/input_ffnn.pkl')
    df_hong = pd.read_csv('tmp/reactivity_database_mapped.csv')
    df_hong_original = pd.read_csv('tmp/hong_data/TrainingSet-2926-PhysOrg.csv')

    df_hong_original.drop(index=[0,1], axis=0, inplace=True)
    df_hong_original.reset_index(inplace=True)
    df_hong_original['index'] = df_hong_original['index'].apply(lambda x: x-2)
    df_hong_original.rename(columns={'index': 'rxn_id'}, inplace=True)

    df_hong_intersection = df_hong_original.loc[df_hong_original.index.isin(df_hong['rxn_id'])]
    logger.info(f'======== 5-fold CV with RF and AdaBoost ========')
    parameters_rf = {'n_estimators': 300, 'max_features': 0.9, 'min_samples_leaf': 1}
    for sample in [25, 50, 100, 200, 300, 400]:
        logger.info(f'Datapoints: {sample}')
        split_dir = f"tmp/cv_hong_{sample}/splits"
        logger.info(f'AdaBoost with 50 descriptors')
        get_cross_val_accuracy_ada_boost_regression(df=df_hong_intersection, logger=logger, n_fold=5, split_dir=split_dir, target_column='Barrier')
        logger.info(f'Model with a learned-VB representation')
        get_cross_val_accuracy_rf_descriptors(df=df_hong_desc, logger=logger, n_fold=5, parameters=parameters_rf, split_dir=split_dir, target_column='DG_TS')
    logger.info(f'AdaBoost with 50 descriptors')
    get_cross_val_accuracy_ada_boost_regression(df=df_hong_intersection, logger=logger, n_fold=5, split_dir='tmp/cv_hong/splits', target_column='Barrier')
    logger.info(f'Model with a learned-VB representation')
    get_cross_val_accuracy_rf_descriptors(df=df_hong_desc, logger=logger, n_fold=5, parameters=parameters_rf, split_dir='tmp/cv_hong/splits', target_column='DG_TS')
    get_cross_val_accuracy_rf_descriptors(df=df_hong_desc, logger=logger_simple, n_fold=5, parameters=parameters_rf, split_dir='tmp/cv_hong/splits', target_column='DG_TS')

    run_train(save_dir='tmp/final_model_hong_4', data_path='tmp/input_ffnn.pkl', target_column='DG_TS', batch_size=32)
    run_train(save_dir='tmp/final_model_hong_1', data_path='tmp/input_ffnn.pkl', target_column='DG_TS', batch_size=32, ensemble_size=1, trained_dir='reactivity_model/results/final_model_1/')
    # Exp.cumyloxyl ^62 Bietti
    run_cv(data_path='tmp/input_ffnn_bietti.pkl', target_column='gibbs_exp', save_dir='tmp/cv_hong_bietti_4', transfer_learning=True, batch_size=5, trained_dir='tmp/final_model_hong_4/', random_state=0)
    mae, rmse, r2 = read_log('tmp/cv_hong_bietti_4/ffn_train.log')
    logger.info("4 ensembles and transfer learning on Bietti Set")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    logger_simple.info('**********BIETTI**********')
    logger_simple.info(f'**4** ensembles and transfer learning on Hong, 10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    update_dataset(file_path=data_manager_file_path, dataset_name='bietti', h=hidden_size_M1, rmse=rmse, mae=mae, r2=r2)
    run_cv(data_path='tmp/input_ffnn_bietti.pkl', target_column='gibbs_exp', save_dir='tmp/cv_hong_bietti_1', transfer_learning=True, batch_size=5, ensemble_size=1, trained_dir='tmp/final_model_hong_1/', random_state=0)
    mae, rmse, r2 = read_log('tmp/cv_hong_bietti_1/ffn_train.log')
    logger.info("1 ensemble and transfer learning on Bietti Set")
    logger.info(f'10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')

    df_bietti = pd.read_pickle('tmp/input_ffnn_bietti.pkl')
    df_bietti.rename(columns={'gibbs_exp':'DG_TS'}, inplace=True)
    df_bietti.to_csv('tmp/input_ffnn_bietti.csv')
    run_reactivity(trained_dir='tmp/final_model_hong_4', target_column='DG_TS', input_file='input_ffnn_bietti.csv')
    df_bietti_pred = pd.read_csv('tmp/pred.csv')
    df_bietti = pd.read_csv('tmp/omega_data/clean_data_omega_exp.csv')
    exp_data_corr(exp_path='tmp/omega_data/clean_data_omega_exp.csv', pred_path='tmp/pred.csv', logger=logger)
    run_reactivity(trained_dir='tmp/final_model_hong_1', target_column='DG_TS', input_file='input_ffnn_bietti.csv', ensemble_size=1)
    df_bietti_pred = pd.read_csv('tmp/pred.csv')
    df_bietti = pd.read_csv('tmp/omega_data/clean_data_omega_exp.csv')
    exp_data_corr(exp_path='tmp/omega_data/clean_data_omega_exp.csv', pred_path='tmp/pred.csv', logger=logger)
    
    # RMechDB Dataset (No additioanl DFT used in this test)
    # TODO implemented by Emilien
    # logger = create_logger('rmechdb_data.log')
    logger.info('********************************')
    logger.info(f'======= RMechDB DATASET M1 hidden_size={hidden_size_M1}=======')
    df_rmechdb = pd.read_pickle('tmp/rmechdb_data/clean_data_RMechDB.pkl')
    create_input_pred(df_rmechdb, 'DG_TS_tunn')
    run_surrogate(chk_path=checkpoint_path)
    logger.info('Surrogate model done')
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS_tunn')
    
    batch_size = 64
    random_state = 0
    ensemble_size = 4
    
    run_cv(batch_size=batch_size, data_path='tmp/input_ffnn.pkl', save_dir='tmp/cv_rmechdb_data', random_state=random_state, ensemble_size=ensemble_size, transfer_learning=True)
    mae, rmse, r2 = read_log('tmp/cv_rmechdb_data/ffn_train.log')
    logger.info(f"{ensemble_size} ensemble(s), batch_size: {batch_size}, random_state: {random_state}")
    logger.info(f'<<pure>> 10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    logger_simple.info('**********RMECHDB**********')
    logger_simple.info(f'<<pure>> **4** ensembles and transfer learning on in-house (hidden size of FFNN is expected to be 2 by Javier, but now 200~230 seems better?), 10-fold CV RMSE, MAE and R^2 for NN with a learned-VB representation: {rmse} {mae} {r2}')
    update_dataset(file_path=data_manager_file_path, dataset_name='rmechdb', h=hidden_size_M1, rmse=rmse, mae=mae, r2=r2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size_M1", default=1200, type=int)
    args = parser.parse_args()
    
    reproduce(args.hidden_size_M1)
