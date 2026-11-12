from typing import List, Union, Tuple
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
from chemprop.models.motif import MotifAttentionPooling, InterfaceEnrichment


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

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

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

        self.stage_split = self.depth // 2   # Split the message passing into two stages: 0 → K-1 and K → D-1, where K = depth // 2

        # Motif-aware representation modules
        self.motif_pooling = MotifAttentionPooling(self.hidden_size, pooling_type="attention")  # "mean" for mean pooling and "attention" for attention pooling
        self.interface_enrichment = InterfaceEnrichment(self.hidden_size, fusion_type="add")


        # Stage III : Bond message re-initialization

        bond_only_fdim = self.bond_fdim - self.atom_fdim

        self.W_stage3 = nn.Linear(
        self.hidden_size + bond_only_fdim,
        self.hidden_size,
        bias=self.bias
        )



    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs with motif-aware mid-encoder abstraction.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).float().to(self.device)

        f_atoms, f_bonds, f_bond_only, w_atoms, w_bonds, a2b, b2a, b2revb, \
        a_scope, b_scope, degree_of_polym = mol_graph.get_components(atom_messages=self.atom_messages)

        f_atoms, f_bonds, f_bond_only, w_atoms, w_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), \
                                                               f_bond_only.to(self.device), w_atoms.to(self.device), w_bonds.to(self.device), \
                                                               a2b.to(self.device), b2a.to(self.device), \
                                                               b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)
        message = self.act_func(input)

        K = self.stage_split  # Midpoint for motif pooling and interface enrichment

        # ===== STAGE I: Message passing (0 → K) =====

        for depth in range(K):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                message = nei_message.sum(dim=1)
            else:
                nei_a_message = index_select_ND(message, a2b)
                nei_a_weight = index_select_ND(w_bonds, a2b)
                nei_a_message = nei_a_message * nei_a_weight[..., None]
                a_message = nei_a_message.sum(dim=1)
                rev_message = message[b2revb]
                message = a_message[b2a] - rev_message

            message = self.W_h(message)
            message = self.act_func(input + message)
            message = self.dropout_layer(message)

        message_K = message.clone()


        # ===== STAGE II: Motif pooling and interface enrichment =====

        # Compute intermediate atom embeddings H^(K)
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message_K, a2x)
        nei_a_weight = index_select_ND(w_bonds, a2x)
        nei_a_message = nei_a_message * nei_a_weight[..., None]
        a_message = nei_a_message.sum(dim=1)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens_K = self.act_func(self.W_o(a_input))

        # Apply motif/interface information
        atom_to_motif = mol_graph.atom_to_motif_batch.to(self.device)
        interface_idx = mol_graph.interface_idx_batch.to(self.device)


        if mol_graph.num_motifs > 0 and atom_to_motif.max().item() >= 0:

            motif_h = self.motif_pooling(
                    atom_hiddens_K,
                    atom_to_motif,
                    mol_graph.num_motifs
            )

            atom_hiddens_K_enriched = self.interface_enrichment(
                atom_hiddens_K,
                motif_h,
                atom_to_motif,
                interface_idx
            )

        else:

            atom_hiddens_K_enriched = atom_hiddens_K



        # ---------------------------------------------------------
        # Update outgoing directed bond messages from interface atoms
        # ---------------------------------------------------------

        stage3_message = message_K.clone()


        interface_mask = torch.zeros(
                atom_hiddens_K_enriched.size(0),
                dtype=torch.bool,
                device=self.device
                        )

        interface_mask[interface_idx] = True

        for bond_idx in range(1, stage3_message.shape[0]):

            source_atom = b2a[bond_idx]

            if not interface_mask[source_atom]:
                continue

            enriched_source = atom_hiddens_K_enriched[source_atom]
            bond_feat = f_bond_only[bond_idx]

            new_input = torch.cat([enriched_source, bond_feat], dim=0)

            semantic_message = self.act_func(
                self.W_stage3(new_input)
            )

            stage3_message[bond_idx] = semantic_message

        stage3_input = stage3_message.clone()

        message = stage3_message


        # ===== STAGE III: Message passing (K → D-1) =====

        for depth in range(K, self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                message = nei_message.sum(dim=1)
            else:
                nei_a_message = index_select_ND(message, a2b)
                nei_a_weight = index_select_ND(w_bonds, a2b)
                nei_a_message = nei_a_message * nei_a_weight[..., None]
                a_message = nei_a_message.sum(dim=1)
                rev_message = message[b2revb]
                message = a_message[b2a] - rev_message

            message = self.W_h(message)
            message = self.act_func(stage3_input + message)
            message = self.dropout_layer(message)

        # ===== Final atom aggregation =====
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)
        nei_a_weight = index_select_ND(w_bonds, a2x)
        nei_a_message = nei_a_message * nei_a_weight[..., None]
        a_message = nei_a_message.sum(dim=1)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)


        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(f'The number of atoms is different from the length of the extra atom features')

            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)
            atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens
                w_atom_vec = w_atoms.narrow(0, a_start, a_size)
                mol_vec = w_atom_vec[..., None] * mol_vec
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / w_atom_vec.sum(dim=0)
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm

                mol_vec = degree_of_polym[i] * mol_vec
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)

        return mol_vecs


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features)
        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages)

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return

        if args.mpn_shared:
            self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
        else:
            self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                                          for _ in range(args.number_of_molecules)])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if type(batch[0]) != BatchMolGraph:
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            # TODO: handle atom_descriptors_batch with multiple molecules per input
            if self.atom_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        atom_features_batch=atom_features_batch,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            elif bond_features_batch is not None:
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            else:
                batch = [mol2graph(b) for b in batch]

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')

            encodings = [enc(ba, atom_descriptors_batch) for enc, ba in zip(self.encoder, batch)]
        else:
            encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]

        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)

        return output
