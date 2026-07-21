"""
Integration test for polymer graph construction.

Tests:
1. Load real polymers from VIPEA dataset
2. MolGraph construction
3. SMARTS motif assignment
4. BatchMolGraph construction

No neural network components.
"""

from pathlib import Path
import pandas as pd

from chemprop.features import set_polymer
from chemprop.features.featurization import MolGraph, BatchMolGraph

# Enable polymer parsing
set_polymer(True)


def load_polymers(n=5):
    dataset = (
        Path(__file__).parent.parent
        / "datasets"
        / "vipea"
        / "chemprop_inputs"
        / "dataset-poly_chemprop.csv"
    )

    df = pd.read_csv(dataset)

    polymers = df["poly_chemprop_input"].head(n).tolist()

    print(f"\n✓ Loaded {len(polymers)} polymers from VIPEA dataset")

    return polymers


def main():

    print("\n" + "=" * 100)
    print("GRAPH CONSTRUCTION TEST")
    print("=" * 100)

    polymers = load_polymers()

    mol_graphs = []

    print("\n" + "-" * 100)
    print("TEST 1 : MolGraph")
    print("-" * 100)

    for i, polymer in enumerate(polymers):

        print(f"\nPolymer {i+1}")
        print(polymer)

        mg = MolGraph(polymer)
        mol_graphs.append(mg)

        print("\nGraph summary")
        print(f"  Atoms              : {mg.n_atoms}")
        print(f"  Directed bonds     : {mg.n_bonds}")
        print(f"  Motif classes      : {mg.num_motifs}")

        if mg.interface_idx is None:
            print("  Interface atoms    : 0")
        else:
            print(f"  Interface atoms    : {len(mg.interface_idx)}")

        if mg.atom_to_motif is not None:

            counts = {}

            for m in mg.atom_to_motif:
                counts[int(m)] = counts.get(int(m), 0) + 1

            print("  Motif distribution")

            for k in sorted(counts):
                print(f"     motif {k}: {counts[k]} atoms")

        assert mg.n_atoms > 0
        assert mg.n_bonds > 0
        assert mg.atom_to_motif is not None

    print("\n" + "-" * 100)
    print("TEST 2 : BatchMolGraph")
    print("-" * 100)

    batch = BatchMolGraph(mol_graphs)

    print("\nBatch summary")

    print(f"  Molecules          : {len(mol_graphs)}")
    print(f"  Total atoms        : {batch.n_atoms}")
    print(f"  Total bonds        : {batch.n_bonds}")
    print(f"  Total motifs       : {batch.num_motifs}")

    print(f"\n  atom_to_motif_batch : {batch.atom_to_motif_batch.shape}")
    print(f"  interface_idx_batch : {batch.interface_idx_batch.shape}")

    print("\na_scope")
    for i, scope in enumerate(batch.a_scope):
        print(f"  Mol {i+1}: {scope}")

    print("\nb_scope")
    for i, scope in enumerate(batch.b_scope):
        print(f"  Mol {i+1}: {scope}")

    assert batch.n_atoms > 0
    assert batch.n_bonds > 0
    assert batch.atom_to_motif_batch is not None
    assert batch.atom_to_motif_batch.shape[0] == batch.n_atoms

    print("\n" + "=" * 100)
    print("✓ GRAPH CONSTRUCTION TEST PASSED")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
