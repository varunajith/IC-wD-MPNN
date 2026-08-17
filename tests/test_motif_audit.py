from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter, deque
import pandas as pd

from chemprop.features import set_polymer
from chemprop.features.featurization import MolGraph

set_polymer(True)


# ------------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------------

INPUT_CSV = (
    Path(__file__).parent.parent
    / "monomer_split_results_analysis"
    / "all_folds_master_analysis.csv"
)

OUTPUT_CSV = "/Users/varun/Documents/polymer-chemprop/monomer_split_results_analysis/master_polymer_descriptors.csv"


# ------------------------------------------------------------------
# Graph distances from interface atoms
# ------------------------------------------------------------------

def shortest_distances(mol, interface_atoms):

    dist = {i: -1 for i in range(mol.GetNumAtoms())}

    q = deque()

    for a in interface_atoms:
        a = int(a)
        dist[a] = 0
        q.append(a)

    while q:

        atom = q.popleft()

        for nb in mol.GetAtomWithIdx(atom).GetNeighbors():

            j = nb.GetIdx()

            if dist[j] == -1:
                dist[j] = dist[atom] + 1
                q.append(j)

    return list(dist.values())


# ------------------------------------------------------------------
# Descriptor extraction
# ------------------------------------------------------------------

def compute_descriptors(polymer):

    mg = MolGraph(polymer)

    motif_sizes = []
    interface_sizes = []
    interface_hetero = []
    interface_aromatic = []

    motif_counter = Counter()

    backbone_atoms = 0

    aromatic_atoms = {
        atom.GetIdx()
        for atom in mg.rwmol.GetAtoms()
        if atom.GetIsAromatic()
    }

    interface_atoms = {int(i) for i in mg.interface_idx}

    for motif_id, atoms in mg.motif_to_atoms.items():

        motif_type = mg.motif_instance_type[motif_id]

        motif_counter[motif_type] += 1

        motif_sizes.append(len(atoms))

        if motif_type == "backbone":
            backbone_atoms += len(atoms)

        contains_interface = any(a in interface_atoms for a in atoms)

        if contains_interface:

            interface_sizes.append(len(atoms))

            hetero = sum(
                mg.rwmol.GetAtomWithIdx(a).GetAtomicNum() not in (1, 6)
                for a in atoms
            )

            aromatic = sum(a in aromatic_atoms for a in atoms)

            interface_hetero.append(hetero)
            interface_aromatic.append(aromatic / len(atoms))

    distances = shortest_distances(mg.rwmol, mg.interface_idx)

    valid = [d for d in distances if d >= 0]

    largest_interface = max(interface_sizes) if interface_sizes else 0

    row = {

        # ---------------- Polymer ----------------

        "n_atoms": mg.n_atoms,
        "n_bonds": mg.n_bonds,
        "n_interface_atoms": len(mg.interface_idx),

        # ---------------- Motifs ----------------

        "motif_count": len(mg.motif_to_atoms),
        "unique_motif_types": len(motif_counter),

        "mean_motif_size":
            sum(motif_sizes) / len(motif_sizes),

        "max_motif_size":
            max(motif_sizes),

        "backbone_atom_fraction":
            backbone_atoms / mg.n_atoms,

        # ---------------- Interface ----------------

        "interface_motif_count":
            len(interface_sizes),

        "interface_mean_size":
            sum(interface_sizes) / max(1, len(interface_sizes)),

        "largest_interface_motif_size":
            largest_interface,

        "interface_mean_heteroatoms":
            sum(interface_hetero) / max(1, len(interface_hetero)),

        "interface_fraction_aromatic":
            sum(interface_aromatic) / max(1, len(interface_aromatic)),

        # ---------------- Graph ----------------

        "mean_distance_interface":
            sum(valid) / len(valid),

        "max_distance_interface":
            max(valid),

        "atoms_within_2_hops":
            sum(d <= 2 for d in valid),

        "atoms_within_3_hops":
            sum(d <= 3 for d in valid),
    }

    # Fixed motif columns

    # Add motif counts dynamically

    for motif, count in motif_counter.items():
        row[f"count_{motif}"] = count
    return row


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    df = pd.read_csv(INPUT_CSV)

    rows = []
    all_motif_columns = set()

    for _, r in df.iterrows():

        print(r["polymer"][:70])

        desc = compute_descriptors(r["polymer"])
        for key in desc:
            if key.startswith("count_"):
                all_motif_columns.add(key)

        # Preserve original information

        for col in [
            "polymer",
            "true_value",
            "baseline_prediction",
            "motif_prediction",
            "baseline_abs_error",
            "motif_abs_error",
            "improvement",
            "relative_improvement",
            "improved",
        ]:
            if col in r.index:
                desc[col] = r[col]

        rows.append(desc)
    
    # Ensure every row has every motif column

    for row in rows:
        for col in all_motif_columns:
            row.setdefault(col, 0)

    out = pd.DataFrame(rows)

    out.to_csv(OUTPUT_CSV, index=False)

    print()
    print("=" * 80)
    print("Saved:", OUTPUT_CSV)
    print("Rows :", len(out))
    print("Cols :", len(out.columns))
    print("=" * 80)


if __name__ == "__main__":
    main()