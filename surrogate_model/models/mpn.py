from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from utilities.featurization import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from utilities.utils_nn import index_select_ND, get_activation_function

class MPNEncoder(nn.Module):
    """
    NOTE NOTE NOTE\n

    atom_features: [one-hot indicating 'C', 'N' or 'O'...] + [one-hot indicating TotalDegree] + [one-hot indicating ExplicitValence] + [one-hot indicating ImplicitValence] + [one-hot indicating FormalCharge] +[0/1 indicating whether Aromatic] + [0/1 indicating if having RadicalElectrons (distinguish mol/rad)]\n

    bond_features: [one-hot indicating None/Single/Double/Triple/Aromatic/] + [0/1 IsConjugated] +[0/1 IsInRing]\n

    input features for MPN:  x_{v}, e_{vw}\n
    Message on bonds::\n
    Initialisation:          h^{0}_{vw} = ReLU( W_i * e_{vw} )\n
    Message Passing:         m^{t+1}_{vw} = \Sigma_{u \in N(v)\w}( h^{t}_{vu} )   Edge's direction is considered.\n
    Information Aggregation: h^{t+1}_{vw} =ReLU( h^{0}_{vw} + W_h * m^{t+1}_{vw} )\n
    Readout for bond:        ReLU( W_o_b * [e_{vw}, h^{T}_{vw}] )\n
    Readout for atom:        ReLU( W_o_a * [x_{v},  \Sigma_{u \in N(v)}(h^{T}_{uv})] )\n
    """

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.use_input_features = args.use_input_features
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        # hidden state readout
        self.W_o_a = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)
        self.W_o_b = nn.Linear(self.bond_fdim + self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, b2br, bond_types, charges, spin_densities = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb, b2br, bond_types = f_atoms.cuda(), f_bonds.cuda(), \
                                                                   a2b.cuda(), b2a.cuda(), b2revb.cuda(), \
                                                                   b2br.cuda(), bond_types.cuda()
            
            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size NOTE h^{0}_{vw}

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden (max_num_bonds: indices of neighbours and padded to max number of bonds going away in whole set)
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden NOTE m^{t+1}_{vw}

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden NOTE h^{t+1}_{vw}

        # atom hidden
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o_a(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # bond hidden
        b_input = torch.cat([f_bonds, message], dim=1)
        bond_hiddens = self.act_func(self.W_o_b(b_input))
        bond_hiddens = self.dropout_layer(bond_hiddens)

        return atom_hiddens, a_scope, bond_hiddens, b_scope, b2br, bond_types, charges, spin_densities  # num_atoms x hidden, remove the first one which is zero padding


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # converts SMILES batch to BatchMolGraph
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(batch, features_batch)

        return output
