#!/usr/bin/python
import pandas as pd
from rdkit import Chem

def create_input_ffnn_hidden(pkl_file, csv_file, target_column=None, additional_data=None, output='input_ffnn'):
    """ we're going to take the csv file of the baseline model and add the line of descriptors pred by the surrogate
    model this surrogate model now predict the charges of open and closed shell molecules and also the dG

    pkl_file (str) : path to file containing output surrogate model

    csv_file (str) : path to file containing input to baseline model

    """

    surrogate_data = pd.read_pickle(pkl_file) # tmp/preds_surrogate_hidden.pkl
    reactivity_data = pd.read_csv(csv_file, index_col=0) # tmp/reactivity_database_mapped.csv

    if additional_data:
        surrogate_additional = pd.read_pickle(additional_data)
        surrogate_data = pd.concat([surrogate_data, surrogate_additional], axis=0, ignore_index=True)
        surrogate_data.drop_duplicates(subset=['smiles'], keep="last", inplace=True)

    mol1_hidden = []
    rad1_hidden = []
    mol2_hidden = []
    rad2_hidden = []
    rad_atom1_hidden = []
    rad_atom2_hidden = []

    for row in reactivity_data.itertuples():
        mol1, rad1, rad2, mol2 = reacs_prods(row.rxn_smiles)
        mol1_info = surrogate_data.loc[surrogate_data.smiles == mol1]
        rad1_info = surrogate_data.loc[surrogate_data.smiles == rad1]
        rad2_info = surrogate_data.loc[surrogate_data.smiles == rad2]
        mol2_info = surrogate_data.loc[surrogate_data.smiles == mol2]

        mol1_hidden.append(mol1_info.mol_hiddens.values[0])
        rad1_hidden.append(rad1_info.mol_hiddens.values[0])
        mol2_hidden.append(mol2_info.mol_hiddens.values[0])
        rad2_hidden.append(rad2_info.mol_hiddens.values[0])
        rad_atom1_hidden.append(rad1_info.rad_atom_hiddens.values[0])
        rad_atom2_hidden.append(rad2_info.rad_atom_hiddens.values[0])

    input_ffnn = pd.DataFrame()

    if 'rxn_id' in reactivity_data.columns:
        input_ffnn['rxn_id'] = reactivity_data.rxn_id
    else:
        input_ffnn['rxn_id'] = reactivity_data.index
    input_ffnn['mol1_hidden'] = mol1_hidden
    input_ffnn['rad1_hidden'] = rad1_hidden
    input_ffnn['mol2_hidden'] = mol2_hidden
    input_ffnn['rad2_hidden'] = rad2_hidden
    input_ffnn['rad_atom1_hidden'] = rad_atom1_hidden
    input_ffnn['rad_atom2_hidden'] = rad_atom2_hidden

    if target_column:
        if target_column in reactivity_data.columns:
            input_ffnn[target_column] = reactivity_data[target_column]

    input_ffnn.to_csv(f'tmp/{output}_hidden.csv')
    input_ffnn.to_pickle(f'tmp/{output}_hidden.pkl')


def reacs_prods(smiles_rxn):
    """ Return the components of the rxn """

    dot_index = smiles_rxn.index('.')
    sdot_index = smiles_rxn.index('.', dot_index + 1)
    limit_index = smiles_rxn.index('>>')
    reac1 = smiles_rxn[:dot_index]
    reac2 = smiles_rxn[dot_index + 1: limit_index]
    prod1 = smiles_rxn[limit_index + 2: sdot_index]
    prod2 = smiles_rxn[sdot_index + 1:]

    return reac1, reac2, prod1, prod2

