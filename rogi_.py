import argparse

from rogi.recall_surrogate import rogi_in_house, rogi_exp_Omega_alkoxy, rogi_Hong_Photoredox, rogi_Omega_Alkoxy_reprod, rogi_omega_bietti_hong, rogi_RMechDB, rogi_tantillo_Cytochrome_P450_reprod

from utils.log import create_logger_emilien


def rogi_hidden(dataset='in_house', args=None):
    '''
    dataset: in_house, omega, omega_exp, hong, hong_bietti, tantillo, atmospheric \n
    fine_tune=True: args will be changed;\n
    fine_tune=False: args not changed regardless of input args.\n
    '''
    logger = create_logger_emilien('rogi_xd.log')
    logger.info(f'=====Hidden size of M1: {args.chk_path_hidden}=====Dataset name: {dataset}=====')
    
    if dataset == 'in_house':
        args.features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
        args.random_state = 0
        # Simple 10-fold CV
        rogi_score = rogi_in_house(args)
            
    if dataset == 'omega':
        args.features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
        args.random_state = 0
        # Pre-selected 60 for test and 238 for train-validation, "10-fold" to change validation set.
        rogi_score = rogi_Omega_Alkoxy_reprod(args)
            
    if dataset == 'omega_exp':
        args.features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
        args.random_state = 0
        # directly test on Exp.Omega using the M2 trained on Omega.
        rogi_score = rogi_exp_Omega_alkoxy(args)
        
    
    if dataset == 'hong':
        args.features = ['rad_atom1_hidden','rad_atom2_hidden']
        args.random_state = 0
        # simple 5-fold CV
        rogi_score = rogi_Hong_Photoredox(args)
            
    if dataset == 'omega_bietti_hong':
        args.features = ['rad_atom1_hidden','rad_atom2_hidden'] # TODO can change
        args.random_state = 0
        # pre-train on Hong and then 10-fold CV on bietti
        # retrain_Hong_Photoredox(args)
        rogi_score = rogi_omega_bietti_hong(args)
            
    if dataset == 'tantillo':
        args.features = ['mol1_hidden', 'rad2_hidden', 'rad_atom2_hidden']
        args.random_state = 0
        # pre-selected 6 for test, randomly select 2 for validation and the left 16 for training from scratch or pre-trained on In-House.
        # retrain_in_house(args)
        rogi_score = rogi_tantillo_Cytochrome_P450_reprod(args)

    if dataset == 'rmechdb':
        args.features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
        args.random_state = 0
        # simple 10-fold CV on RMechDB (training from scratch or pre-training on In-House)
        rogi_score = rogi_RMechDB(args)
        
    logger.info(rogi_score)
        
    return rogi_score


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='in_house', type=str)
    parser.add_argument("--features", nargs="+", type=str,  default=['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden'])
    parser.add_argument("--random_state", default=0, type=int)
    parser.add_argument("--chk_path", default="surrogate_model/qmdesc_wrap/model.pt", type=str)
    parser.add_argument("--chk_path_hidden", default=1200, type=int, help='hidden size of M1')
    args = parser.parse_args()

    args.chk_path = f"surrogate_model/output_h{args.chk_path_hidden}_b50_e100/model_0/model.pt"
    
    rogi_score = rogi_hidden(dataset=args.dataset, args=args)
    
    