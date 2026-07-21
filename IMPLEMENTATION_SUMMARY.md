# Motif-Aware WD-MPNN Implementation Summary

**Version:** 4.0 — Implementation Complete, Experimental Evaluation Ongoing  
**Date:** 20th July 2026

---

# Overview

This project extends Aldeghi et al.'s weighted directed message passing neural network (WD-MPNN) with a latent motif abstraction stage inserted midway through message propagation.

The original stochastic polymer graph, weighted directed message passing formulation and graph topology are preserved. Motifs are **not** introduced as graph nodes; instead they are learned as latent semantic representations from contextual atom embeddings and injected only into polymerization interface atoms before message passing resumes.

---

# Current Pipeline

```
Polymer Repeat Unit
        │
        ▼
MolGraph + Motif Decomposition
        │
        ▼
Stage I WD-MPNN
(Local directed message passing)
        │
        ▼
Contextual Atom Embeddings
        │
        ▼
Attention-based Motif Pooling
        │
        ▼
Motif Embeddings
        │
        ▼
Interface Atom Enrichment
        │
        ▼
Regenerate Outgoing Interface Messages
        │
        ▼
Stage III WD-MPNN
        │
        ▼
Readout
        │
        ▼
Property Prediction
```

---

# Representation Hierarchy

The model learns two successive abstractions.

```
Directed edge messages
        │
        ▼
Contextual atom embeddings
(local directional integration)
        │
        ▼
Motif-conditioned interface atoms
(higher-order structural semantics)
        │
        ▼
Regenerated directed messages
        │
        ▼
Continue WD-MPNN
```

Motifs exist only as latent semantic representations and never become graph nodes.

---

# Core Design Principles

- Preserve Aldeghi's stochastic polymer graph.
- Preserve weighted directed message passing.
- Learn motif representations from contextual atom embeddings.
- Modify only polymerization interface atoms.
- Resume completely standard WD-MPNN after semantic injection.

---

# Implementation Status

## Representation

✓ Priority-based motif decomposition

✓ SMARTS functional-group detection

✓ Aromatic-system detection (including fused systems)

✓ Backbone decomposition

✓ Interface atom detection

✓ MolGraph / BatchMolGraph motif storage

---

## Network

✓ Stage I WD-MPNN

✓ Attention-based motif pooling

✓ Learnable motif embeddings

✓ Interface enrichment

✓ Semantic projection

✓ Outgoing interface message regeneration

✓ Stage III WD-MPNN continuation

✓ End-to-end training and inference

---

# Validation

Verified:

- Correct motif batching
- Correct interface identification
- Correct tensor dimensions
- Correct outgoing message replacement
- Stable forward/backward propagation
- Successful end-to-end training

---

# Experimental Progress

Completed

✓ Baseline Aldeghi reproduction

✓ Motif-aware WD-MPNN benchmarking

✓ Nine-fold monomer-level evaluation

✓ Random split evaluation

✓ Prediction comparison across all folds

✓ Overall performance analysis

✓ Error distribution analysis

✓ Difficulty-based improvement analysis

✓ Initial ablation studies
- Atom refresh only
- Zero-motif injection

Current aggregate results:

- ~45% RMSE reduction
- ~46% MAE reduction
- ~72% of polymers improved relative to baseline

Largest improvements occur on polymers that are most difficult for the baseline model.

---

# Current Interpretation

Current evidence suggests two complementary mechanisms contribute to performance:

1. **Interface refresh**

   Updating polymerization interface atoms alone improves performance over the original WD-MPNN, indicating that selective interface refinement itself provides a useful inductive bias.

2. **Motif semantics**

   Learned motif embeddings provide additional higher-order structural information that further improves prediction accuracy beyond interface refresh alone.

Understanding the relative contribution of these mechanisms is the current focus of the project.

---

# Remaining Work

## Ablations

- Complete remaining folds for current ablations (if required)
- Compare atom refresh vs motif enrichment quantitatively
- Decide whether additional controls (e.g. random motif vectors) are necessary

## Chemical Analysis

- Characterize chemistry of the most improved polymers
- Characterize chemistry of the most degraded polymers
- Identify motif-level enrichment trends
- Select representative qualitative case studies

## Manuscript

- Consolidate experimental figures
- Finalize discussion of architectural mechanisms
- Prepare publication

---

# Current Research Milestone

The project has progressed from an architectural proposal to a validated representation learning framework.

Implementation is complete.

Benchmarking demonstrates substantial improvements over the Aldeghi baseline.

The remaining research effort is no longer architectural development but understanding **why** motif-aware message passing improves polymer property prediction through ablation studies and chemical interpretation.

---

# Resume Here

Current priority:

1. Analyze chemistry of Top-100 improved vs Top-100 degraded polymers.
2. Interpret the atom-refresh ablation relative to the full motif model.
3. Finalize remaining manuscript figures and begin writing.