"""
Motif-aware representation learning for wD-MPNN.

Modules for computing motif embeddings and enriching interface atoms.
"""

import torch
import torch.nn as nn


class MotifAttentionPooling(nn.Module):
    """
    Attention-based pooling to compute motif embeddings from atom embeddings.
    
    Given atom embeddings and their motif assignments, computes a learnable
    attention-weighted representation for each motif.
    """
    """
    Supports:
    - "attention": learnable attention-weighted pooling
    - "mean": simple mean pooling
    """


    def __init__(self, hidden_size: int, pooling_type: str = "attention"):
        """
        :param hidden_size: Dimensionality of atom/motif embeddings.
        :param pooling_type: Pooling method ("attention" or "mean").
        """
        super(MotifAttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.pooling_type = pooling_type

        # Learnable attention weights for pooling
        if self.pooling_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False),
            )

        elif self.pooling_type == "mean":
            self.attention = None

        else:
            raise ValueError(
                f"Unknown pooling_type: {self.pooling_type}. "
                f"Expected 'attention' or 'mean'."
            )


    def forward(self,
                atom_hiddens: torch.Tensor,
                atom_to_motif: torch.Tensor,
                num_motifs: int,
                ) -> torch.Tensor:
        """
        Compute motif embeddings via attention or mean pooling.

        :param atom_hiddens: Atom embeddings, shape (num_atoms, hidden_size)
        :param atom_to_motif: Motif assignment per atom, shape (num_atoms,), dtype=LongTensor
        :param num_motifs: Total number of distinct motifs
        :return: Motif embeddings, shape (num_motifs, hidden_size)
        """
        
        # Initialize motif embeddings
        motif_hiddens = torch.zeros(num_motifs, self.hidden_size, device=atom_hiddens.device)

        # Attention pooling
        if self.pooling_type == "attention":
        # Compute attention logits
            attn_logits = self.attention(atom_hiddens)  # (num_atoms, 1)
        
        # Gather atoms by motif and apply attention or mean pooling
        for motif_id in range(num_motifs):
            mask = (atom_to_motif == motif_id)
            if not mask.any():
                continue
            
            motif_atoms = atom_hiddens[mask]  # (motif_size, hidden_size)
            if self.pooling_type == "attention":

                attention_scores = attn_logits[mask]    # (motif_size, 1)
                attention_weights = torch.softmax(attention_scores, dim=0)  # (motif_size, 1)
                # Attention Weighted pooling
                motif_hiddens[motif_id] = (attention_weights * motif_atoms).sum(dim=0)  # (hidden_size,)

            elif self.pooling_type == "mean":
                motif_hiddens[motif_id] = motif_atoms.mean(dim=0)

        
        return motif_hiddens


class InterfaceEnrichment(nn.Module):
    """
    Enrich interface atom embeddings with their corresponding motif embeddings.
    
    Only polymerization interface atoms are modified by fusing their representations
    with the motif representation to which they belong.
    """

    def __init__(self, hidden_size: int, fusion_type="mlp"):
        """
        :param hidden_size: Dimensionality of atom/motif embeddings.
        :param fusion_type: Type of fusion to use ("mlp" or "add").
        """
        super(InterfaceEnrichment, self).__init__()
        self.hidden_size = hidden_size
       
        self.fusion_type = fusion_type
        if self.fusion_type == "mlp":
            self.fusion = nn.Sequential(
                                nn.Linear(2 * hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size)
                                    )
    def forward(self,
                atom_hiddens: torch.Tensor,
                motif_hiddens: torch.Tensor,
                atom_to_motif: torch.Tensor,
                interface_idx: torch.Tensor) -> torch.Tensor:
        """
        Enrich interface atoms with motif information.

        :param atom_hiddens: Atom embeddings, shape (num_atoms, hidden_size)
        :param motif_hiddens: Motif embeddings, shape (num_motifs, hidden_size)
        :param atom_to_motif: Motif assignment per atom, shape (num_atoms,), dtype=LongTensor
        :param interface_idx: Indices of interface atoms, shape (num_interface,), dtype=LongTensor
        :return: Updated atom embeddings with enriched interface atoms, shape (num_atoms, hidden_size)
        """
        updated_hiddens = atom_hiddens.clone()
        
        if interface_idx.numel() == 0:
            print(f"[InterfaceEnrichment] No interface atoms. atom_hiddens: {atom_hiddens.shape} (unchanged)")
            return updated_hiddens
        
        # For each interface atom, retrieve its motif embedding and fuse
        for intf_atom_idx in interface_idx:
            motif_id = atom_to_motif[intf_atom_idx]
            atom_emb = atom_hiddens[intf_atom_idx]      # (hidden_size,)
            motif_emb = motif_hiddens[motif_id]         # (hidden_size,)

            if self.fusion_type == "mlp":
            # Concatenate and project
                fused = torch.cat([atom_emb, motif_emb], dim=0)  # (hidden_size*2,)
                enriched = self.fusion(fused)                    #(hidden_size,)
                updated = atom_emb + enriched 

            elif self.fusion_type == "add":
                updated = atom_emb + motif_emb

            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
            
            updated_hiddens[intf_atom_idx] = updated

        
        return updated_hiddens
