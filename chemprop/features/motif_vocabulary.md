# Motif Vocabulary Specification (Version 1.0)

## Project

Motif-Aware Weighted Directed Message Passing Neural Network (WD-MPNN)

Author: Varun Ajith

---

# Purpose

This document defines the polymer motif vocabulary used by the Motif-Aware
WD-MPNN.

The vocabulary is **not intended to discover new chemistry**. Instead, it
provides a deterministic decomposition of each polymer repeat unit into
chemically meaningful motif instances.

These motif instances serve as the semantic units over which Stage II performs
attention-based pooling before continuing the original Aldeghi weighted directed
message passing.

This document therefore forms the **formal specification of the representation**.

---

# Design Principles

The motif vocabulary follows five principles.

1. Every motif must correspond to a chemically meaningful substructure.

2. Every motif should have a well-defined electronic or structural role in
polymer properties.

3. Every atom belongs to exactly one motif instance.

4. Motifs are predefined using established chemistry rather than learned.

5. Remaining atoms not belonging to predefined motifs are grouped into
connected backbone fragments.

---

# Representation Pipeline

For every polymer repeat unit

Polymer Graph

↓

Detect ring systems

↓

Detect functional groups

↓

Assign motif instances

↓

Group remaining connected atoms into backbone fragments

↓

Attention pooling over atoms belonging to each motif

↓

Motif embedding

↓

Broadcast motif context back to atoms

↓

Continue Aldeghi WD-MPNN message passing

---

# Motif Categories

## Category A : Ring Systems

These motifs are detected using RDKit ring perception rather than SMARTS.

Reason:

Aromatic and heteroaromatic rings behave as delocalized electronic systems and
should therefore be represented by a single embedding.

| Motif | Detection | Priority | Notes |
|-------|-----------|----------|------|
| Aromatic carbocyclic ring | RDKit RingInfo | Highest | Benzene, fused phenyl systems |
| Heteroaromatic ring | RDKit RingInfo | Highest | Pyridine, quinoline, thiophene, imidazole etc.|

---

## Category B : Carbonyl Chemistry

Detected using SMARTS.

Priority order is important because several motifs overlap.

Priority

Ester

↓

Amide

↓

Imide

↓

Carboxylic acid

↓

Acid chloride

↓

Remaining carbonyl

| Motif | Detector | Motivation |
|-------|----------|-----------|
| Ester | SMARTS | PET, PMMA |
| Amide | SMARTS | Nylon |
| Imide | SMARTS | Polyimides |
| Carboxylic acid | SMARTS | Functional polymers |
| Acid chloride | SMARTS | Polymer precursors |
| Carbonyl | SMARTS | Remaining ketones/aldehydes |

---

## Category C : Oxygen Chemistry

| Motif | Detector |
|-------|----------|
| Ether | SMARTS |
| Alcohol | SMARTS |

---

## Category D : Nitrogen Chemistry

| Motif | Detector |
|-------|----------|
| Amine | SMARTS |
| Nitrile | SMARTS |
| Nitro | SMARTS |

---

## Category E : Sulfur Chemistry

Sulfur environments are treated separately because of their distinct electronic
effects.

| Motif | Detector |
|-------|----------|
| Thioether | SMARTS |
| Sulfoxide | SMARTS |
| Sulfone | SMARTS |

---

## Category F : Halogen Chemistry

Rather than treating halogens as isolated atoms, the bonded carbon environment
is represented.

| Motif | Detector |
|-------|----------|
| C–F | SMARTS |
| C–Cl | SMARTS |
| C–Br | SMARTS |
| C–I | SMARTS |

---

## Category G : Backbone Fragments

After all predefined motifs have been assigned,

remaining atoms are grouped into connected components.

Each connected component forms one backbone motif instance.

Detection

Graph traversal

Reason

Backbone flexibility strongly influences polymer properties while preserving
local connectivity.

---

# Motif Instance Construction

A motif type may appear multiple times within one polymer.

Example

PET repeat unit

Phenyl

Ether

Backbone

Ether

Carbonyl

Phenyl

This produces six motif instances although only four motif types are present.

Stage II pools atoms **within each motif instance**, not across all motifs of
the same type.

---

# Interface Atoms

Polymer attachment atoms are **not motif types**.

Instead they are stored as a separate annotation.

Interface atoms are later used during Stage III interface-aware enrichment.

---

# Representation Data Structures

Each MolGraph stores

atom_to_motif

Maps every atom to one motif instance.

motif_to_atoms

Dictionary mapping motif instance to its constituent atoms.

motif_instance_type

Maps each motif instance to its chemical identity.

Example

Motif 0

type = aromatic_ring

atoms = [0,1,2,3,4,5]

Motif 1

type = ester

atoms = [6,7,8]

Motif 2

type = backbone

atoms = [9,10]

---

# Attention Pooling

For every motif instance

Atom embeddings

↓

Self-attention

↓

Weighted aggregation

↓

Motif embedding

The resulting motif embedding is then broadcast back to all atoms belonging to
that motif before continuing Stage III message passing.



---

# Future Extensions

Possible future motif categories

• Boronic esters

• Phosphates

• Siloxanes

• Organometallic motifs

• Data-driven motif discovery

These extensions should not require modification of the network architecture.

Only the motif vocabulary would expand.