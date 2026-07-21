# `math_implementation_spec.md`

# Version 1: Mid-Encoder Motif Abstraction for WD-MPNN

---

# Design Contract

The following architectural assumptions are fixed and **must not** be modified during implementation.

* Original atomic graph topology is preserved.
* Aldeghi's stochastic transition probabilities remain unchanged.
* Message passing before and after the midpoint is unchanged.
* Motifs are **not** graph nodes.
* No hierarchical graph or hypergraph is introduced.
* Motif embeddings are computed exactly once.
* Only polymerization interface atoms are enriched.
* Polymer readout remains unchanged.

The only intervention is a semantic abstraction layer inserted midway through the encoder.

---

# 1. Stage I — Original WD-MPNN (Inherited)

Let

$$
G=(V,E)
$$

denote the original polymer graph.

Let the total message-passing depth be

$$
D,
$$

and define the midpoint

$$
K=\left\lfloor\frac{D}{2}\right\rfloor .
$$

The first (K) message-passing iterations follow Aldeghi *without modification*, producing the intermediate atom embeddings

$$
H^{(K)}
=======

\left{
\mathbf h_1^{(K)},
\mathbf h_2^{(K)},
\ldots,
\mathbf h_N^{(K)}
\right}.
$$

At this point the embeddings encode rich local chemistry while remaining less entangled with the global polymer environment than the final encoder representations.

### Code intervention

**None.**

Execution proceeds exactly as in `MPNEncoder.forward()` until iteration (K).

---

# 2. Deterministic Motif Assignment

Prior to training, each repeat unit is decomposed into chemically meaningful motifs using a deterministic decomposition procedure (SMARTS, BRICS or equivalent).

Define

$$
\phi:V\rightarrow{1,\ldots,M},
$$

where

$$
\phi(i)
$$

returns the motif containing atom (v_i).

Each motif is therefore

$$
M_m
===

\left{
v_i
\mid
\phi(i)=m
\right}.
$$

The polymerization interface atoms are extracted directly from the dataset parser,

$$
I
=

{
i_1,\ldots,i_P
}.
$$

Both mappings remain fixed throughout training.

### New metadata

| Tensor          | Shape            |
| --------------- | ---------------- |
| `atom_to_motif` | `(N_atoms,)`     |
| `interface_idx` | `(N_interface,)` |

### Code intervention

Extend

* `MolGraph`
* `BatchMolGraph`

to store these lookup tensors.

---

# 3. Motif Representation Learning (New)

Using the intermediate atom embeddings,

$$
H^{(K)},
$$

the embeddings belonging to motif (M_m) are gathered as

$$
\mathcal H_m
============

\left{
\mathbf h_i^{(K)}
\mid
\phi(i)=m
\right}.
$$

A learnable attention pooling operator computes a single motif representation,

$$
\boxed{
\mathbf z_m
===========

\operatorname{AttnPool}
\left(
\mathcal H_m
\right)
}
$$

Collecting all motifs,

$$
Z
=

{
\mathbf z_1,
\mathbf z_2,
\ldots,
\mathbf z_M
}.
$$

The motif embeddings are computed **once** and never updated again. They represent a semantic snapshot of recurring chemical motifs before continued message passing increasingly entangles their identity with the surrounding polymer.

### Tensor evolution

$$
H^{(K)}
\longrightarrow
Z
$$

### Code intervention

**Input**

* `atom_h`
* `atom_to_motif`

**Output**

* `motif_h`

**New module**

`MotifPooling`

**Insertion point**

Immediately after layer

```python
depth // 2
```

inside `MPNEncoder.forward()`.

---

# 4. Interface Enrichment (New)

Each interface atom retrieves the embedding of the motif to which it belongs,

$$
\mathbf z_{\phi(i)},
\qquad
i\in I.
$$

Its hidden representation is then enriched,

$$
\boxed{
\tilde{\mathbf h}_i^{(K)}
=========================

\operatorname{Enrich}
\left(
\mathbf h_i^{(K)},
\mathbf z_{\phi(i)}
\right),
\qquad
i\in I.
}
$$

All remaining atoms remain unchanged,

$$
\tilde{\mathbf h}_i^{(K)}
=========================

\mathbf h_i^{(K)},
\qquad
i\notin I.
$$

The updated hidden representation is therefore

$$
\tilde H^{(K)}
==============

\left{
\tilde{\mathbf h}_1^{(K)},
\ldots,
\tilde{\mathbf h}_N^{(K)}
\right}.
$$

Only the interface atom embeddings are modified. The motif embeddings are then discarded.

### Information flow

$$
H^{(K)}
\longrightarrow
Z
\longrightarrow
\tilde H^{(K)}
$$

Unlike hierarchical GNNs, motifs never communicate with one another and never participate in subsequent message passing.

### Code intervention

**Input**

* `atom_h`
* `motif_h`
* `atom_to_motif`
* `interface_idx`

**Output**

* `updated_atom_h`

**New module**

`InterfaceEnrichment`

---

# 5. Stage II — Original WD-MPNN (Inherited)

The enriched representation replaces the original midpoint representation,

$$
H^{(K)}
;\Rightarrow;
\tilde H^{(K)}.
$$

The remaining encoder layers proceed exactly as in Aldeghi,

$$
\tilde H^{(K)}
\rightarrow
H^{(K+1)}
\rightarrow
\cdots
\rightarrow
H^{(D)}.
$$

No modifications are introduced to

* message passing,
* transition probabilities,
* residual updates,
* polymer readout,
* prediction head.

The enriched interface representation simply becomes the hidden state refined by the remainder of the original WD-MPNN.

---

# 6. Overall Tensor Evolution

$$
G
\rightarrow
H^{(K)}
\rightarrow
Z
\rightarrow
\tilde H^{(K)}
\rightarrow
H^{(D)}
\rightarrow
h_{\mathrm{poly}}
\rightarrow
\hat y
$$

---

# 7. Mapping to Aldeghi

| Aldeghi Stage                          | Version 1           |
| -------------------------------------- | ------------------- |
| Graph construction                     | Unchanged           |
| Edge message passing                   | Unchanged           |
| Atom embedding computation             | Unchanged           |
| Intermediate atom embeddings (H^{(K)}) | **Insertion point** |
| Polymer readout                        | Unchanged           |

---

# 8. Summary of Code Interventions

| Stage             | New Object                       | Code Addition                |
| ----------------- | -------------------------------- | ---------------------------- |
| Preprocessing     | `atom_to_motif`, `interface_idx` | Extend `MolGraph`            |
| Mid Encoder       | (Z)                              | `MotifPooling`               |
| Mid Encoder       | (\tilde H^{(K)})                 | `InterfaceEnrichment`        |
| Remaining Encoder | None                             | Resume original `MPNEncoder` |

---

## Implementation Pipeline

```text
Graph
   │
   ▼
WD-MPNN (0 → K)
   │
   ▼
H^(K)
   │
   ▼
MotifPooling
   │
   ▼
Motif Embeddings (Z)
   │
   ▼
InterfaceEnrichment
   │
   ▼
H~^(K)
   │
   ▼
WD-MPNN (K → D)
   │
   ▼
Polymer Readout
   │
   ▼
Prediction
```

**Core hypothesis:** Rather than allowing motif identity to remain implicitly distributed across atomic embeddings throughout the encoder, Version 1 explicitly captures a transferable motif representation at mid-depth and injects this semantic information through the polymerization interface before allowing the original WD-MPNN to continue unchanged. This preserves the strengths of Aldeghi's architecture while introducing a single, controlled semantic abstraction layer.
