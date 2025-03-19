from rogi import rogi_xd
from rogi.rogi_xd_utils import IntegrationDomain
import pandas as pd
import numpy as np
from utils.input_for_pred import create_input_pred
from emilien.utils_hidden import predict_hidden
from utils.create_input_ffnn_hidden import create_input_ffnn_hidden
from utils.run_models import run_surrogate
from utils.create_input_ffnn import create_input_ffnn
from data_manager import update_dataset_rogi
from utils.tantillo.final_functions import add_pred_tantillo_hidden, add_pred_tantillo

def get_data_hidden(target, args):
    '''
    target: 
    
    in-house: 'dG_act_corrected'
    
    omega: 'G_act'
    
    selectivity:
    
    Hong:
    
    bietti:
    
    tantillo:
    
    rmechdb:
    '''
    
    df_hidden = pd.read_pickle('tmp/input_ffnn_hidden.pkl')
    
    y_hidden = df_hidden[target].to_numpy()
    
    vector_features = []
    for feature in args.features:
        vector_features.append(np.stack(df_hidden[feature].values))
    x_hidden = np.concatenate(vector_features, axis=1)
    
    return x_hidden, y_hidden

def get_data_desc(target, features=['dG_forward', 'dG_reverse', 'q_reac0', 'qH_reac0', 'q_reac1', 's_reac1', 'q_prod0', 's_prod0', 'q_prod1', 'qH_prod1', 'BV_reac1', 'BV_prod0', 'fr_dG_forward', 'fr_dG_reverse']):
    
    df_desc = pd.read_pickle('tmp/input_ffnn.pkl')
    
    y_desc = df_desc[target].to_numpy()
    x_desc = df_desc[features].to_numpy()
    
    return x_desc, y_desc


def get_rogi(x_hidden, y_hidden, x_desc, y_desc, args):
    print('shape of X_hidden:', x_hidden.shape)
    
    rogi_xd_score_hidden = rogi_xd.rogi(x=x_hidden, y=y_hidden) 
    
    update_dataset_rogi(file_path='rogi_xd_hidden.pkl', dataset_name=args.dataset, h=args.chk_path_hidden, rogi_score=rogi_xd_score_hidden)
    
    print('shape of X_desc:', x_desc.shape)
    
    rogi_xd_score_desc = rogi_xd.rogi(x=x_desc, y=y_desc) 
    
    update_dataset_rogi(file_path='rogi_xd_desc.pkl', dataset_name=args.dataset, h=args.chk_path_hidden, rogi_score=rogi_xd_score_desc)
    
    rogi_xd_score = {'hidden':rogi_xd_score_hidden, 'desc':rogi_xd_score_desc}
    
    return rogi_xd_score


def rogi_in_house(args):
    
    df = pd.read_csv('tmp/own_dataset/reactivity_database_corrected.csv', index_col=0)
    create_input_pred(df=df, target_column='dG_act_corrected', target_column_2='dG_rxn')
    predict_hidden(chk_path=args.chk_path)
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/reactivity_database_mapped.csv', 'dG_act_corrected')
    
    x_hidden, y_hidden = get_data_hidden(target='dG_act_corrected', args=args)
    
    #==========================================
    
    df = pd.read_csv('tmp/own_dataset/reactivity_database_corrected.csv', index_col=0)
    create_input_pred(df=df, target_column='dG_act_corrected', target_column_2='dG_rxn')
    run_surrogate(chk_path=args.chk_path)
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'dG_act_corrected')
    
    x_desc, y_desc = get_data_desc(target='dG_act_corrected')
    
    rogi_xd_score = get_rogi(x_hidden, y_hidden, x_desc, y_desc, args)
    
    return rogi_xd_score

def rogi_Omega_Alkoxy_reprod(args):
    
    predict_hidden(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv', chk_path=args.chk_path)
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/omega_data/clean_data_omega.csv', 'G_act')

    df_omega = pd.read_pickle('tmp/input_ffnn_hidden.pkl')
    df_omega = df_omega.loc[df_omega['G_act'] != 'FALSE']
    df_omega.to_pickle('tmp/input_ffnn_hidden.pkl')
    
    x_hidden, y_hidden = get_data_hidden(target='G_act', args=args)
    
    #==========================================
    
    run_surrogate(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv', chk_path=args.chk_path)
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/omega_data/clean_data_omega.csv', 'G_act', 'tmp/omega_data/additional_data_omega.pkl')
    df_omega = pd.read_pickle('tmp/input_ffnn.pkl')
    df_omega = df_omega.loc[df_omega['G_act'] != 'FALSE']
    df_omega.to_pickle('tmp/input_ffnn.pkl')
    
    x_desc, y_desc = get_data_desc(target='G_act')
    
    rogi_xd_score = get_rogi(x_hidden, y_hidden, x_desc, y_desc, args)
    
    return rogi_xd_score

def rogi_exp_Omega_alkoxy(args):
    
    predict_hidden(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv', chk_path=args.chk_path)
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/omega_data/clean_data_omega_selectivity.csv', 'gibbs', output='input_ffnn_selectivity')
    
    df_selectivity = pd.read_pickle('tmp/input_ffnn_selectivity_hidden.pkl')
    df_selectivity.rename(columns={'gibbs': 'G_act'}, inplace=True)
    df_selectivity.to_pickle('tmp/input_ffnn_hidden.pkl')
    
    x_hidden, y_hidden = get_data_hidden(target='G_act', args=args)
    
    run_surrogate(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv', chk_path=args.chk_path)
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/omega_data/clean_data_omega_selectivity.csv', 'gibbs', 'tmp/omega_data/additional_data_omega.pkl', 'input_ffnn_selectivity')
    
    df_selectivity = pd.read_pickle('tmp/input_ffnn_selectivity.pkl')
    df_selectivity.rename(columns={'gibbs': 'G_act'}, inplace=True)
    df_selectivity.to_pickle('tmp/input_ffnn.pkl')
    
    x_desc, y_desc = get_data_desc(target='G_act')
    
    rogi_xd_score = get_rogi(x_hidden, y_hidden, x_desc, y_desc, args)
    
    return rogi_xd_score

def rogi_Hong_Photoredox(args):
    
    df_hong = pd.read_csv('tmp/hong_data/training_hong_clean.csv', index_col=0)
    create_input_pred(df_hong, 'DG_TS')
    predict_hidden(chk_path=args.chk_path)
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS')
    
    x_hidden, y_hidden = get_data_hidden(target='DG_TS', args=args)
    
    
    df_hong = pd.read_csv('tmp/hong_data/training_hong_clean.csv', index_col=0)
    create_input_pred(df_hong, 'DG_TS')
    run_surrogate(chk_path=args.chk_path)
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS')
    
    x_desc, y_desc = get_data_desc(target='DG_TS')
    
    rogi_xd_score = get_rogi(x_hidden, y_hidden, x_desc, y_desc, args)
    
    return rogi_xd_score

def rogi_omega_bietti_hong(args):
    
    predict_hidden(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv', chk_path=args.chk_path)
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/omega_data/clean_data_omega_exp.csv', 'gibbs_exp')
    
    x_hidden, y_hidden = get_data_hidden(target='gibbs_exp', args=args)
    
    run_surrogate(test_file='tmp/omega_data/species_reactivity_omega_dataset.csv', chk_path=args.chk_path)
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/omega_data/clean_data_omega_exp.csv', 'gibbs_exp', 'tmp/omega_data/additional_data_omega.pkl', output='input_ffnn_bietti')
    
    df_bietti = pd.read_pickle('tmp/input_ffnn_bietti.pkl')
    df_bietti.to_pickle('tmp/input_ffnn.pkl')
    
    x_desc, y_desc = get_data_desc(target='gibbs_exp')
    
    rogi_xd_score = get_rogi(x_hidden, y_hidden, x_desc, y_desc, args)
    
    return rogi_xd_score

def rogi_tantillo_Cytochrome_P450_reprod(args):
    
    #Only calculate rogi of training set
    
    predict_hidden(test_file='tmp/tantillo_data/species_reactivity_tantillo_dataset.csv', chk_path=args.chk_path)
    add_pred_tantillo_hidden(train_file='tmp/tantillo_data/clean_data_tantillo.csv',
                            test_file='tmp/tantillo_data/clean_data_steroids_tantillo.csv', 
                            pred_file='tmp/preds_surrogate_hidden.pkl')
    
    df_tantillo = pd.read_pickle('tmp/tantillo_data/input_tantillo_hidden.pkl')
    df_tantillo.to_pickle('tmp/input_ffnn_hidden.pkl')
    
    x_hidden, y_hidden = get_data_hidden(target='DFT_Barrier', args=args)
    
    
    run_surrogate(test_file='tmp/tantillo_data/species_reactivity_tantillo_dataset.csv', chk_path=args.chk_path)
    add_pred_tantillo(train_file='tmp/tantillo_data/clean_data_tantillo.csv',
                    test_file='tmp/tantillo_data/clean_data_steroids_tantillo.csv', 
                    pred_file='tmp/preds_surrogate.pkl')
    
    df_tantillo = pd.read_pickle('tmp/tantillo_data/input_tantillo.pkl')
    df_tantillo.to_pickle('tmp/input_ffnn.pkl')
    
    x_desc, y_desc = get_data_desc(target='DFT_Barrier', features=['s_rad', 'Buried_Vol'])
    
    rogi_xd_score = get_rogi(x_hidden, y_hidden, x_desc, y_desc, args)
    
    return rogi_xd_score

def rogi_RMechDB(args):
    
    df_rmechdb = pd.read_pickle('tmp/rmechdb_data/clean_data_RMechDB.pkl')
    create_input_pred(df_rmechdb, 'DG_TS_tunn')
    predict_hidden(chk_path=args.chk_path)
    create_input_ffnn_hidden('tmp/preds_surrogate_hidden.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS_tunn')
    
    x_hidden, y_hidden = get_data_hidden(target='DG_TS_tunn', args=args)
    
    df_rmechdb = pd.read_pickle('tmp/rmechdb_data/clean_data_RMechDB.pkl')
    create_input_pred(df_rmechdb, 'DG_TS_tunn')
    run_surrogate(chk_path=args.chk_path)
    create_input_ffnn('tmp/preds_surrogate.pkl', 'tmp/reactivity_database_mapped.csv', 'DG_TS_tunn')
    
    x_desc, y_desc = get_data_desc(target='DG_TS_tunn')
    
    rogi_xd_score = get_rogi(x_hidden, y_hidden, x_desc, y_desc, args)
    
    return rogi_xd_score