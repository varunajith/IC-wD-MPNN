# Motif-aware WD-MPNN (Version 1)

## Implementation Design Specification

---

# Objective

Introduce an explicit **motif representation layer** into the existing WD-MPNN architecture while preserving Aldeghi's original polymer representation and stochastic message passing framework.

The goal is to evaluate the contribution of an explicit semantic representation with **minimal architectural modifications**.

---

# Design Constraints

The following components remain **unchanged** from the original implementation.

* MolGraph construction
* BatchMolGraph
* Atomic graph representation
* Directed message passing
* Transition probabilities
* Stoichiometric weighting
* Polymer readout
* Prediction head

The only architectural additions are

* Motif Assignment
* Motif Pooling
* Interface Fusion

---

# Core Representation

The model continues to operate on the original atomic graph

[
G=(V,E)
]

where

* nodes are atoms
* edges are directed bonds

No motif graph, hypergraph or hierarchical graph is introduced.

A motif is simply defined as

[
M_k \subseteq V
]

i.e. a chemically meaningful subset of atoms in the existing graph.

---

# Representation Pipeline

```
Input Polymer

↓

MolGraph

↓

BatchMolGraph

↓

WD-MPNN

↓

Contextualized Atom Embeddings

↓

Motif Module

↓

Motif-aware Interface Atoms

↓

Remaining WD-MPNN

↓

Weighted Polymer Readout

↓

Prediction
```

---

# Stage I — Local Chemical Encoding

The first half of the WD-MPNN encoder is left unchanged.

Purpose

* learn contextualized atom embeddings
* encode local chemical environments

Output

```
Atom Embeddings
```

---

# Motif Module

The motif module introduces the only new latent representation.

Input

```
Contextualized Atom Embeddings
```

Operations

```
Atom Assignment

↓

Pooling

↓

Motif Embeddings
```

The motif embedding is computed once and remains fixed for the remainder of the forward pass.

No message passing is performed on motifs.

---

# Interface Fusion

Each polymerization attachment atom

```
[*:1]

[*:2]

...
```

is associated with its corresponding motif.

The interface atom embedding is enriched by fusing it with the motif embedding.

Conceptually

```
Interface Atom

+

Motif Embedding

↓

Motif-aware Interface Atom
```

Graph connectivity is unchanged.

Only the feature representation is updated.

---

# Semantic Abstraction Depth

The motif module is inserted **midway through the WD-MPNN encoder**.

Define

* (D) = total message passing depth
* (k) = semantic abstraction depth

Version 1 fixes

[
k=\left\lfloor\frac{D}{2}\right\rfloor
]

This divides the encoder into

```
Local Chemical Encoding

↓

Motif Abstraction

↓

Polymer Contextualization
```

The midpoint is treated as a principled engineering approximation and can later be explored as an ablation parameter.

---

# Forward Pipeline

```
Atomic Features

↓

WD-MPNN Layers (0 → D/2)

↓

Contextualized Atom Embeddings

↓

Motif Assignment

↓

Motif Pooling

↓

Motif Embeddings

↓

Interface Fusion

↓

WD-MPNN Layers (D/2 → D)

↓

Weighted Polymer Readout

↓

Prediction
```

---

# Codebase Modifications

Only localized changes should be required.

## Existing Components

No changes should be made to

* MolGraph
* BatchMolGraph
* Dataset parsing
* Transition probability handling
* Readout
* Predictor

## Modified Component

`MPNEncoder`

The encoder should be divided into two message passing stages.

```
for layer in range(depth):

    message_passing()

    if layer == depth // 2:

        motif_module()

        interface_fusion()
```

The remaining message passing layers continue exactly as in the original implementation.

---

# Scope of Version 1

Version 1 intentionally excludes

* hierarchical graphs
* motif message passing
* motif attention
* graph coarsening
* graph rewiring
* learned motif discovery

The purpose of Version 1 is solely to evaluate the effect of introducing an explicit motif representation within the existing WD-MPNN framework.

---

# Guiding Principle

The proposed architecture does **not** replace the atomic representation.

Instead, it inserts a semantic abstraction layer between

```
Local Chemical Encoding

and

Polymer-level Contextualization
```

while preserving the original stochastic polymer representation introduced by Aldeghi et al.


Likely files to modify

chemprop/models/mpn.py
    - Split MPNEncoder into two stages.
    - Insert motif module at depth // 2.

chemprop/features/featurization.py (or equivalent)
    - Create motif assignments from monomer graphs.
    - Store atom → motif mapping.

chemprop/models/motif.py (new)
    - Motif pooling.
    - Interface fusion.

Everything else remains unchanged.