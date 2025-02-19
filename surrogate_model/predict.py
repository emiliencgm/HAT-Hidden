"""Loads a trained model checkpoint and makes predictions on a dataset."""
from utilities.parsing import parse_predict_args
from run.make_predictions import make_predictions
from utilities.utils_gen import load_args

import pandas as pd
from rdkit import Chem
import numpy as np
import time

def num_atoms_bonds(smiles, explicit_Hs):
    m = Chem.MolFromSmiles(smiles)

    if explicit_Hs:
        m = Chem.AddHs(m)

    return len(m.GetAtoms()), len(m.GetBonds())

if __name__ == '__main__':
    '''
    NOTE NOTE NOTE
    Load the args used for M1 prediction and within M1 training (generally the same thing?).
    train_args.atom_targets = ['spin_densities', 'charges_all_atom']
    train_args.bond_targets = []
    train_args.mol_targets  = [['Buried_Vol', 'dG', 'frozen_dG']]
    selection of descriptors is shown in the paper Fig.2.

    Read args.test_path: tmp/species_reactivity_dataset.csv, which are mapped input reactions (actually mols and rads), and input into the trained M1.
    M1's prediction is made and then split into atom-level or molecule-level QM descriptor values.
    Prediction saved in tmp/preds_surrogate.pkl

    <tmp/splited_test_preds.csv> (created by myself)

    smiles,spin_densities,charges_all_atom,Buried_Vol,dG,frozen_dG

    [CH3:4][CH2:5],
    "[-0.1731127   0.9921864  -0.03579919 -0.03579919 -0.03579919 -0.1178854 -0.1178854 ]",
    "[-0.3649907  -0.35060567  0.13506104  0.13506104  0.13506104  0.15520664 0.15520664]",
    0.1830801318377435,
    91.21993788119445,
    136.43066121224427
    '''
    args = parse_predict_args()
    train_args = load_args(args.checkpoint_paths[0])

    if not hasattr(train_args, 'single_mol_tasks'): 
            train_args.single_mol_tasks = False

    test_df = pd.read_csv(args.test_path, index_col=0)
    smiles = test_df.smiles.tolist() 
    
    start = time.time() 
    test_preds, test_smiles = make_predictions(args, smiles=smiles)
    end = time.time()

    print('time:{}s'.format(end-start))

    train_a_targets = train_args.atom_targets
    train_b_targets = train_args.bond_targets
    train_m_targets = train_args.mol_targets
    if train_args.single_mol_tasks:
        train_m_targets= [item for sublist in train_m_targets for item in sublist]
    n_atoms, n_bonds = zip(*[num_atoms_bonds(x, train_args.explicit_Hs) for x in smiles]) 
    # n_atoms = (7, 7, 8, 11, 9, 12, 8, 8)

    df = pd.DataFrame({'smiles': smiles})

    for idx, target in enumerate(train_a_targets):
        props = test_preds[idx]
        props = np.split(props.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
        df[target] = props

    n_a_targets = len(train_a_targets)

    for idx, target in enumerate(train_b_targets):
        props = test_preds[idx+n_a_targets]
        props = np.split(props.flatten(), np.cumsum(np.array(n_bonds)))[:-1]
        df[target] = props

    n_ab_targets = len(train_a_targets) + len(train_b_targets)

    for idx, target in enumerate(train_m_targets):
        props = test_preds[idx+n_ab_targets]
        df[target] = props
    
    df.to_pickle(args.preds_path)
    df.to_csv('tmp/splited_test_preds.csv')