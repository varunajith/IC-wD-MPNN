"""
Visualize motif assignments for polymers.

Creates a PNG grid where every polymer is coloured according to the
motif decomposition performed inside MolGraph.
"""
import sys
from pathlib import Path
import math
import random
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from chemprop.features import set_polymer
from chemprop.features.featurization import MolGraph
from collections import Counter

set_polymer(True)


# ---------------------------------------------------------
# Load polymers
# ---------------------------------------------------------

def load_polymers():

    df = pd.read_csv(
        "/Users/varun/Documents/polymer-chemprop/monomer_split_results_analysis/difficulty_chemistry_analysis/easy_decile.csv"
    )

    df["original_index"] = df.index
     # Smallest absolute improvements first
    df = df.sort_values(
        by="improvement",
        ascending=True
    ).reset_index(drop=True)

    # Keep only the top 50
    df = df.head(50)

    return df

# ---------------------------------------------------------
# Random colour palette
# ---------------------------------------------------------

random.seed(42)

# ---------------------------------------------------------
# Motif colour palette
# ---------------------------------------------------------

import colorsys

BASE_COLOURS = {
    "aromatic_ring": (0.15, 0.35, 0.95),   # Blue
    "alcohol":       (0.15, 0.75, 0.25),   # Green
    "fluoro":        (0.95, 0.20, 0.20),   # Red
    "nitrile":       (0.65, 0.25, 0.85),   # Purple
    "ester":         (1.00, 0.60, 0.00),   # Orange
    "amide":         (0.10, 0.75, 0.75),   # Cyan
    "ether":         (0.95, 0.85, 0.15),   # Yellow
    "carbonyl":      (0.80, 0.40, 0.10),   # Brown
    "amine":         (1.00, 0.40, 0.70),   # Pink
    "sulfur":        (0.80, 0.80, 0.20),   # Olive
    "backbone":      (0.55, 0.55, 0.55),   # Grey
}


def motif_colour(motif_name, instance_idx):
    """
    Same motif type -> same base colour.
    Different instances -> different shades.
    """

    base = BASE_COLOURS.get(motif_name, (0.45, 0.45, 0.45))

    h, l, s = colorsys.rgb_to_hls(*base)

    # create different shades
    offsets = [-0.18, -0.09, 0.00, 0.09, 0.18, 0.27]

    l = max(0.25, min(0.85, l + offsets[instance_idx % len(offsets)]))

    return colorsys.hls_to_rgb(h, l, s)


# ---------------------------------------------------------
# Draw one polymer
# ---------------------------------------------------------

def draw_polymer(mg, title, size=(600, 500)):

    mol = Chem.Mol(mg.rwmol)

# ---------------------------------------------------------
# Label interface atoms
# ---------------------------------------------------------

    r_lookup = {}

# Read R labels from the original RWMol
    for atom in mg.rwmol.GetAtoms():

        if atom.HasProp("R"):

            r_lookup[atom.GetIdx()] = atom.GetProp("R")

# Attach labels to the DRAWING molecule
    for atom in mol.GetAtoms():

        idx = atom.GetIdx()

        if idx in r_lookup:

            atom.SetProp(
            "atomNote",
            f"{idx}\n{r_lookup[idx]}"
        )

        else:

            atom.SetProp(
            "atomNote",
            str(idx)
        )
    highlight_atoms = []
    atom_colours = {}

    highlight_bonds = []
    bond_colours = {}

    #
    # atoms
    #

    instance_counter = {}

    for motif_id, atoms in mg.motif_to_atoms.items():

        motif_name = mg.motif_instance_type[motif_id]

        shade = instance_counter.get(motif_name, 0)

        c = motif_colour(motif_name, shade)

        instance_counter[motif_name] = shade + 1

        atom_set = set(atoms)

        for atom in atoms:

            highlight_atoms.append(atom)

            atom_colours[atom] = c

        #
        # bonds
        #

        for bond in mol.GetBonds():

            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()

            if a in atom_set and b in atom_set:

                highlight_bonds.append(bond.GetIdx())

                bond_colours[bond.GetIdx()] = c
    
        # ---------------------------------------------------------
    # Highlight polymer interface atoms
    # ---------------------------------------------------------

    INTERFACE_COLOUR = (1.0, 0.84, 0.0)   # Gold

    for atom in mg.interface_idx:

        atom = int(atom)

        # Override motif colour
        atom_colours[atom] = INTERFACE_COLOUR

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])

    opts = drawer.drawOptions()

    atom_radii = {}

    for atom in highlight_atoms:
        atom_radii[atom] = 0.35

    for atom in mg.interface_idx:
        atom_radii[int(atom)] = 0.45

    rdMolDraw2D.PrepareAndDrawMolecule(
    drawer,
    mol,
    legend=title,
    highlightAtoms=highlight_atoms,
    highlightAtomColors=atom_colours,
    highlightAtomRadii=atom_radii,
    highlightBonds=highlight_bonds,
    highlightBondColors=bond_colours,
)

    drawer.FinishDrawing()

    from io import BytesIO

    return Image.open(BytesIO(drawer.GetDrawingText()))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

# -----------------------------------------
# Dataset-wide motif statistics
# -----------------------------------------

motif_instance_counts = Counter()
motif_atom_counts = Counter()

def main():

    df = load_polymers()

    images = []
    summary_rows = []

    for i, row in df.iterrows():

        print(f"Drawing polymer {i+1}")

        polymer = row["polymer"]

        improvement = row["improvement"]
        baseline = row["baseline_prediction"]
        motif = row["motif_prediction"]
        truth = row["true_value"]

        mg = MolGraph(polymer)
# -----------------------------------------
# Collect motif statistics
# -----------------------------------------

        for motif_id, atoms in mg.motif_to_atoms.items():

            motif_name = mg.motif_instance_type[motif_id]
            if motif_name == "aromatic_ring" and len(atoms) != 6:
                print(
                        f"[Polymer {i+1}] "
                        f"Ring instance {motif_id}: "
                        f"{len(atoms)} atoms -> {atoms}"
                )

            motif_instance_counts[motif_name] += 1

            motif_atom_counts[motif_name] += len(atoms)
        
        summary = Counter(
                        mg.motif_instance_type.values()
                    )
        
        monomer_string = polymer.split("|")[0]

        monomers = monomer_string.split(".")

        if len(monomers) >= 2:
            monomer1 = monomers[0]
            monomer2 = monomers[1]
        else:
            monomer1 = monomer_string
            monomer2 = ""

        summary_rows.append({
            "original_index": row["original_index"],
            "rank": i + 1,
            "improvement": improvement,
            "true_value": truth,
            "baseline_prediction": baseline,
            "motif_prediction": motif,
            "monomer1": monomer1,
            "monomer2": monomer2,
            "interface_atoms": len(mg.interface_idx),
            "motif_count": len(mg.motif_to_atoms),
            "motif_summary": dict(summary),
        })

        legend = (
            f"Rank {i+1}\n"
            f"Δ={improvement:.3f}\n"
            f"GT={truth:.3f}\n"
            f"B={baseline:.3f}\n"
            f"M={motif:.3f}\n"
             + ", ".join(
            f"{k}×{v}"
            for k, v in summary.items()
    )
)

        img = draw_polymer(
             mg,
            legend
        )

        images.append(img)

    #
    # Make grid
    #

    cols = 2

    rows = math.ceil(len(images) / cols)

    w, h = images[0].size

    canvas = Image.new(
        "RGB",
        (cols * w, rows * h),
        "white",
    )

    for i, img in enumerate(images):

        x = (i % cols) * w

        y = (i // cols) * h

        canvas.paste(img, (x, y))

    Path("motif_visualisation_easy_decile_bottom50_all_folds").mkdir(exist_ok=True)

    outfile = Path("motif_visualisation_easy_decile_bottom50_all_folds") / "motif_visualization.png"

    canvas.save(outfile)
    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
                    Path("motif_visualisation_easy_decile_bottom50_all_folds") / "polymer_summary.csv",
                    index=False,
    )

    print("\n" + "=" * 80)
    print("MOTIF VOCABULARY STATISTICS")
    print("=" * 80)

    print(f"{'Motif':20s} {'Instances':>10s} {'Atoms':>10s} {'Avg size':>10s}")
    print("-" * 80)

    for motif in sorted(motif_instance_counts):

        n_instances = motif_instance_counts[motif]
        n_atoms = motif_atom_counts[motif]
        avg_size = n_atoms / n_instances if n_instances > 0 else 0

        print(
        f"{motif:20s}"
        f"{n_instances:10d}"
        f"{n_atoms:10d}"
        f"{avg_size:10.2f}"
    )
    
    covered_atoms = sum(motif_atom_counts.values())

    print(f"\nTotal atoms assigned to motifs : {covered_atoms}")
    print(f"Backbone motif instances       : {motif_instance_counts.get('backbone',0)}")

    print("=" * 80)

    print()
    print("=" * 80)
    print("Saved:", outfile)
    print(
         "Saved:",
         Path("motif_visualisation_easy_decile_bottom50_all_folds") / "polymer_summary.csv",
    )
    print("=" * 80)


if __name__ == "__main__":

    main()