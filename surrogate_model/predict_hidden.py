"""Loads a trained model checkpoint and makes predictions on a dataset."""
from utilities.parsing import parse_predict_args
from run.make_predictions_hidden import make_predictions_hidden
from utilities.utils_gen import load_args

from rdkit import Chem
import pandas as pd
import time
import numpy as np

def num_atoms_bonds(smiles, explicit_Hs):
    m = Chem.MolFromSmiles(smiles)

    if explicit_Hs:
        m = Chem.AddHs(m)

    return len(m.GetAtoms()), len(m.GetBonds())

def get_rad_index(smiles):
    """ Get the index of the radical atom"""

    mol = Chem.MolFromSmiles(smiles)
    flag = 0

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1:
            idx = atom.GetIdx()
            flag = 1
    if flag:
        return idx
    else:
        return None

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
    smiles = test_df.smiles.tolist() #species并非按照反应顺序排序mol1、rad2、rad1、mol2的……！
    
    start = time.time() 
    test_preds_hidden, test_smiles = make_predictions_hidden(args, smiles=smiles)
    end = time.time()
    print('time:{}s'.format(end-start))

    # TODO planed to save test_preds_hiddens as:
    # rxn_id | mol1 | atom_losing_H | rad2   | atom_getting_H | rad1  | atom_lost_H | mol2 | atom_got_H
    # CCO.C[CH2]>>CC[O].CC
    # 0      | CCO  |       O       | C[CH2] |       C        | CC[O] |      O      |  CC  |     C
    
    
    n_atoms, n_bonds = zip(*[num_atoms_bonds(x, train_args.explicit_Hs) for x in smiles]) 
    # n_atoms = (7, 7, 8, 11, 9, 12, 8, 8)
    rad_atom_hiddens = []
    cumulate_atom_idx = 0
    for idx, mol_smiles in enumerate(smiles):
         atom_rad_idx = get_rad_index(mol_smiles)
         if atom_rad_idx != None:
            atom_rad_idx += cumulate_atom_idx
            rad_atom_hiddens.append(np.array(test_preds_hidden[0][atom_rad_idx], dtype=float))
         else:
            rad_atom_hiddens.append(np.array([0.0]))
         cumulate_atom_idx += n_atoms[idx]

    df = pd.DataFrame({'smiles': smiles, 'mol_hiddens': list(test_preds_hidden[1]), 'rad_atom_hiddens': rad_atom_hiddens})              

    df.to_pickle(args.preds_path) # tmp/preds_surrogate_hidden.pkl
    df.to_csv('tmp/splited_test_preds_hidden.csv')

    