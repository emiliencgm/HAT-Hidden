from typing import List

import torch
import torch.nn as nn
from tqdm import trange
import numpy as np

from utilities.data import MoleculeDataset, StandardScaler


def predict(model: nn.Module,
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

    preds = []

    num_iters, iter_step = len(data), batch_size
    smiles_batch_all = []

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        with torch.no_grad():
            batch_preds = model(batch, features_batch)
            # spin_densities: num_atoms * 1
            # charges_all_atom: num_atoms * 1
            # Buried_Vol, dG, frozen_dG: num_mols (is 4, 2 reactants, 2 products) * 3 

        batch_preds = [x.data.cpu().numpy() for x in batch_preds]

        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds, atom_types = mol_batch.get_atom_types())
            # predicted desc are inverse-scaled. Because desc within training set is scaled.

        # Collect vectors
        preds.append(batch_preds)
        smiles_batch_all.extend(smiles_batch)
    preds = [np.concatenate(x) for x in zip(*preds)]

    return preds, smiles_batch_all
