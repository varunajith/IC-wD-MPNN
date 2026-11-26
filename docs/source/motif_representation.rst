Motif representation
====================

Purpose
-------

The motif representation provides an explicit description of recurring
chemical substructures in a polymer input. It complements the
atom-level representation learned by directed message passing.

Motifs are auxiliary representations. They are not converted into nodes of
a separate hierarchical graph. Instead, motif information is pooled and
used to condition atom representations at polymerization interfaces.

Vocabulary and assignment
-------------------------

The motif vocabulary and SMARTS-based assignment logic are implemented in
:mod:`chemprop.features.motif_vocabulary`. The vocabulary includes
chemically meaningful categories such as aromatic systems, alcohols,
amides, amines, carbonyls, halogen-containing groups, esters, ethers,
nitriles, sulfur-containing groups and a backbone category.

Category-specific matching and prioritisation are used to handle
overlapping SMARTS matches consistently.

Pooling and interface enrichment
--------------------------------

Motif pooling and interface enrichment are implemented in
:mod:`chemprop.models.motif_representation`. At the intermediate
message-passing representation, atom-level information is aggregated
according to motif assignments. The resulting motif information is used
to condition representations of polymerization-interface atoms.

Integration
-----------

The encoder in :mod:`chemprop.models.mpn` applies motif pooling and
interface enrichment after the initial message-passing stage. Enriched
interface representations are then used to initialise the subsequent
message-passing stage. This allows motif-level context to influence
message passing while retaining the atomic graph as the underlying
computational graph.
