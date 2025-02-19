from argparse import Namespace

import torch
import torch.nn as nn

from .mpn import MPN
from .ffn import MultiReadout
from utilities.utils_nn import initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self):
        """
        Initializes the MoleculeModel.
        """
        super(MoleculeModel, self).__init__()


    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        # Create readout layers (several FFN)
        self.readout = MultiReadout(args, args.atom_targets, args.bond_targets, args.mol_targets,
                                    args.atom_constraints, args.bond_constraints)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output_all = self.readout(self.encoder(*input))
        # spin_densities: num_atoms * 1
        # charges_all_atom: num_atoms * 1
        # Buried_Vol, dG, frozen_dG: num_mols (=4, 2 reactants, 2 products) * 3 
        return output_all
    
    #TODO NOTE Changed by CGM
    def forward_hidden_space(self, *input):
        """
        Return the hidden representation instead of the predicted descriptors.
        Return atom_hiddens, because bond_hiddens are None (No bond target given)
        """
        atom_hiddens, a_scope, bond_hiddens, b_scope, b2br, bond_types, charges, spin_densities = self.encoder(*input)
        # input into MPN: <f_atoms> = nearly one-hot with 42 dim; <f_bonds> = [f_atoms_start/end, 0/1 indicators] = 42+7=49 dim.
        # NOTE 在输入（以bond为中心的消息传递的，atom_messages=False的）MPN的特征中，<f_bonds>的前几个维度是<f_atoms>，后面才是描述该键的特征。首先这说明键是有方向的，比如前几维是起点原子的特征。这样在MPN中初始化message（h^{0}_{vw}）时，才能通过W_i的映射获得该键更多的信息——键的性质不止受本身的类型（单双键）影响，还受成键原子的影响。
        # NOTE 为什么hidden space中"原子数"33？The position [index 0] is **zero padding** for f_atoms and f_bonds.
        # zero padding那一维是用于满足charges和spin_densities的constraints的（使用另一个layer学得给各个原子charge加权的权重，甚至会使用attention），当不用预测这两个性质时，舍弃第一维即可。NOTE 从索引1开始才是对应的原子的输出。
        # NOTE f_* 输入model中得到 *_hiddens
        # f_atoms & atom_hiddens: (1 + num_atom) * hidden_size (=1200), 1 for zero padding
        # f_bonds & bond_hiddens: (1 + 2*num_bond) * hidden_size (=1200), 这里X2是因为原始7维的f_bonds前面被加上了起点和终点的原子的42维属性：[f_atoms_start, 0/1 indicators] & [f_atoms_end, 0/1 indicators]
        # NOTE learnable weights and attention are used after FFN. Consider semantic deviation of MPN caused by FFN+attention (and also constraints by charges and spins, 为什么知道真实的charge还要预测一个charge并用真实值修正？？？两个charge有概念的不同？).
        # NOTE 为什么model(MPN)/BatchMolGraph被调用了两次
        """
        hidden space for every atom. Index starts from 0. The zero padding is removed.
        """
        self.atom_hiddens = atom_hiddens[1:]

        """
        hidden space for every mol (4 mols in total).
        """
        scope = a_scope
        _,b = atom_hiddens.shape
        hidden = torch.empty((len(a_scope),b), dtype=torch.float64)
        for i, (a_start, a_size) in enumerate(scope):
            mol_a_hidden = atom_hiddens[a_start:a_start+a_size]
            # NOTE mean of all the atoms is used to readout mol_target Buried_Vol, dG and frozen_dG.
            mol_hidden = torch.mean(mol_a_hidden, 0)
            hidden[i] = mol_hidden
        self.mol_hiddens = hidden

        return self.atom_hiddens, self.mol_hiddens

def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    args.output_size = 1

    model = MoleculeModel()
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
