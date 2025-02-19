from typing import List

import torch
import torch.nn as nn
from tqdm import trange
import numpy as np

from utilities.data import MoleculeDataset, StandardScaler


def predict_hidden(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None
            ) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.

    NOTE NOTE NOTE

    An inverse-scale transformation is done after the prediction (if scaler is used within model) to restore the output.
    """
    model.eval()

    preds_hidden = []

    num_iters, iter_step = len(data), batch_size
    smiles_batch_all = []

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        with torch.no_grad():
            batch_preds_hidden = model.forward_hidden_space(batch, features_batch) #TODO extract hidden space of prediction
            # batch_preds_hidden = atom_hiddens, mol_hiddens within only one batch
            # atom_hiddens: num_atoms * hidden_size
            # mol_hiddens:  num_mol (=4) * hidden_size

        batch_preds_hidden = [x.data.cpu().numpy() for x in batch_preds_hidden]

        # Collect vectors
        preds_hidden.append(batch_preds_hidden)
        smiles_batch_all.extend(smiles_batch)
    
    preds_hidden = [np.concatenate(x) for x in zip(*preds_hidden)] # NOTE 按列分组并将整列cat，这保证了输出中将不同batch的atom信息拼接在一起，不同batch的mol信息也拼接在一起。
    #preds_hidden = [(a1, m1), 
    #                (a2, m2)] 
    #             --> zip(*) --> 第一列a1和a2被同时枚举，然后cat；第二列m1和m2同理
    # = [atom_all, mol_all] = [ [atom_hiddens_batch_1 : atom_hiddens_batch_2], [mol_hiddens_batch_1 : mol_hiddens_batch_2] ] = [ (70+70)*1200, (8+8)*1200 ]
    
    return preds_hidden, smiles_batch_all
