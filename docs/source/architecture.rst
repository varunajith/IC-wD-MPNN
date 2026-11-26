Model architecture
==================

Overview
--------

IC-wD-MPNN is a motif-aware directed message-passing neural network for
polymer property prediction. It operates on the original atom-level graph
and introduces motif information into message passing at polymerization
interfaces.

Message-passing stages
----------------------

The encoder in :mod:`chemprop.models.mpn` follows a staged computation:

1. **Initial message passing:** directed messages propagate over the atomic
   graph to construct atom representations.
2. **Motif-aware interface conditioning:** at an intermediate
   representation, motif assignments are pooled and used by the
   interface-enrichment operation to update representations associated with
   polymerization-interface atoms.
3. **Further message passing:** enriched interface representations
   initialise subsequent messages, allowing motif-conditioned information
   to propagate through the atomic graph.
4. **Readout:** final atom representations are aggregated into a
   polymer-level representation for property prediction.

The motif module is inserted after the first
:math:`\lfloor D/2 \rfloor` message-passing layers, where :math:`D` is the
message-passing depth.

Main components
---------------

The model comprises polymer-aware graph construction, directed message
passing, motif assignment and pooling, interface enrichment, polymer-level
readout and a feed-forward prediction network.

The motif vocabulary is implemented in
:mod:`chemprop.features.motif_vocabulary`; motif pooling and interface
enrichment are implemented in :mod:`chemprop.models.motif_representation`.
Their integration into message passing is implemented in
:mod:`chemprop.models.mpn`.
