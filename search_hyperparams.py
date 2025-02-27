import itertools
import json
import pandas as pd
from main_emilien import reproduce_hidden
import argparse

def search_best_hyperparams(dataset, chk_path_hidden):
    '''
    chk_path_hidden: int, hidden_size of M1
    '''
    # hyperparams that we search for 
    # layers = [0, 1, 2]
    layers = [0, 1]
    # hidden_sizes = [128, 256, 512, 768, 1024, 1536, 2048]
    hidden_sizes = [128, 256]

    results = []
    
    # fixed hyperparams
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
    
    args.dataset = dataset
    args.chk_path_hidden = chk_path_hidden
    args.fine_tune = 1
    args.chk_path = f"surrogate_model/output_h{chk_path_hidden}_b50_e100/model_0/model.pt"
    args.random_state = 0
    args.lr = 0.0277
    args.dropout = 0.0
    args.transfer_learning = False # reproduce_hidden() changes this automatically
    
    if dataset in ['in_house', 'omega', 'omega_exp', 'rmechdb']:
        args.features = ['mol1_hidden', 'rad1_hidden', 'mol2_hidden', 'rad2_hidden', 'rad_atom1_hidden','rad_atom2_hidden']
    elif dataset in ['hong', 'omega_bietti_hong']:
        args.features = ['rad_atom1_hidden','rad_atom2_hidden']
    elif dataset in ['tantillo']:
        args.features = ['mol1_hidden', 'rad2_hidden', 'rad_atom2_hidden']
    else:
        return
    
    for layer, hidden_size in itertools.product(layers, hidden_sizes):
        args.layers = layer
        args.hidden_size = hidden_size
        
        rmse, mae, r2 = reproduce_hidden(dataset=dataset, args=args, fine_tune=True)
        
        results.append({
            "dataset": dataset,
            "chk_path_hidden": chk_path_hidden,
            "layers": layer,
            "hidden_size": hidden_size,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
    
    df = pd.DataFrame(results)
    
    # look for best hyperparams
    best_rmse = df.loc[df['rmse'].idxmin()]
    best_mae = df.loc[df['mae'].idxmin()]
    best_r2 = df.loc[df['r2'].idxmax()]
    
    best_params = {
        "best_rmse": best_rmse.to_dict(),
        "best_mae": best_mae.to_dict(),
        "best_r2": best_r2.to_dict()
    }
    
    # save best hyperparams as JSON
    with open(f"best_params_{dataset}_{chk_path_hidden}.json", "w") as f:
        json.dump(best_params, f, indent=4)
    
    # save all results during running
    df.to_csv(f"results_{dataset}_{chk_path_hidden}.csv", index=False)
    
    return best_params


if __name__ == '__main__':  
    
    dataset_list = ['in_house', 'omega', 'omega_exp', 'hong', 'omega_bietti_hong', 'tantillo', 'rmechdb']
    chk_path_hidden_list = list(range(100, 2001, 100))
    
    for chk_path_hidden in chk_path_hidden_list:
        for dataset in dataset_list:
            search_best_hyperparams(dataset=dataset, chk_path_hidden=chk_path_hidden)