"""
polymer_motifs.py

Central repository of chemically meaningful polymer motifs used by the
Motif-Aware WD-MPNN.

Each motif definition corresponds to a MOTIF TYPE.

During graph construction, every occurrence of a motif becomes a
separate MOTIF INSTANCE.

Example
-------
PET repeat unit

Phenyl -- Ester -- Phenyl

produces

Instance 0 : aromatic_ring
Instance 1 : ester
Instance 2 : aromatic_ring

The decomposition algorithm in featurization.py iterates through this
vocabulary in descending priority order.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class PolymerMotif:
    """
    Definition of a polymer motif type.
    """

    name: str

    # Higher priority motifs claim atoms first.
    priority: int

    # Detector type
    #
    # "smarts"
    # "rdkit_ring"
    # "backbone"
    detector: str

    # SMARTS string if applicable.
    smarts: Optional[str] = None

    # Documentation only.
    description: str = ""

    # Example polymers containing this motif.
    examples: Optional[List[str]] = None


###############################################################################
# Functional motifs
###############################################################################

FUNCTIONAL_MOTIFS = [

    ###########################################################################
    # Carbonyl chemistry
    ###########################################################################

    PolymerMotif(
        name="ester",
        priority=100,
        detector="smarts",
        smarts="[#6](=O)-[#8]-[#6]",
        description="Ester linkage",
        examples=["PET", "PMMA"]
    ),

    PolymerMotif(
        name="amide",
        priority=95,
        detector="smarts",
        smarts="[#6](=O)-[#7]",
        description="Amide linkage",
        examples=["Nylon"]
    ),

    PolymerMotif(
        name="imide",
        priority=94,
        detector="smarts",
        smarts=None,          # TODO: validate SMARTS
        description="Imide group",
        examples=["Polyimide"]
    ),

    PolymerMotif(
        name="carboxylic_acid",
        priority=93,
        detector="smarts",
        smarts=None,
        description="Carboxylic acid",
    ),

    PolymerMotif(
        name="acid_chloride",
        priority=92,
        detector="smarts",
        smarts=None,
        description="Acid chloride",
    ),

    PolymerMotif(
        name="carbonyl",
        priority=90,
        detector="smarts",
        smarts="[#6](=O)",
        description="Remaining carbonyl groups"
    ),

    ###########################################################################
    # Oxygen chemistry
    ###########################################################################

    PolymerMotif(
        name="ether",
        priority=85,
        detector="smarts",
        smarts="[OX2]([#6])[#6]",
        description="Ether linkage",
        examples=["PEEK"]
    ),

    PolymerMotif(
    name="alcohol",
    priority=80,
    detector="smarts",
    smarts="[OX2H]",
    description="Hydroxyl oxygen"
    ),

    ###########################################################################
    # Nitrogen chemistry
    ###########################################################################

    PolymerMotif(
        name="amine",
        priority=75,
        detector="smarts",
        smarts="[NX3]",
        description="Amine"
    ),

    PolymerMotif(
        name="nitrile",
        priority=74,
        detector="smarts",
        smarts="[#6]#[#7]",
        description="Nitrile",
        examples=["PAN"]
    ),

    PolymerMotif(
        name="nitro",
        priority=73,
        detector="smarts",
        smarts=None,
        description="Nitro group"
    ),

    ###########################################################################
    # Sulfur chemistry
    ###########################################################################

    PolymerMotif(
        name="thioether",
        priority=70,
        detector="smarts",
        smarts=None,
        description="Thioether"
    ),

    PolymerMotif(
        name="sulfoxide",
        priority=69,
        detector="smarts",
        smarts=None,
        description="Sulfoxide"
    ),

    PolymerMotif(
        name="sulfone",
        priority=68,
        detector="smarts",
        smarts=None,
        description="Sulfone"
    ),

    ###########################################################################
    # Halogen chemistry
    ###########################################################################

    PolymerMotif(
        name="c_f",
        priority=60,
        detector="smarts",
        smarts=None,
        description="Carbon-fluorine bond"
    ),

    PolymerMotif(
        name="c_cl",
        priority=59,
        detector="smarts",
        smarts=None,
        description="Carbon-chlorine bond"
    ),

    PolymerMotif(
        name="c_br",
        priority=58,
        detector="smarts",
        smarts=None,
        description="Carbon-bromine bond"
    ),

    PolymerMotif(
        name="c_i",
        priority=57,
        detector="smarts",
        smarts=None,
        description="Carbon-iodine bond"
    ),


    PolymerMotif(
    name="fluoro",
    priority=72,
    detector="smarts",
    smarts="[#9]"
),

    PolymerMotif(
    name="chloro",
    priority=72,
    detector="smarts",
    smarts="[#17]"
),

    PolymerMotif(
    name="bromo",
    priority=72,
    detector="smarts",
    smarts="[#35]"
),

    PolymerMotif(
    name="iodo",
    priority=72,
    detector="smarts",
    smarts="[#53]"
),

]

###############################################################################
# Ring motifs
###############################################################################

RING_MOTIFS = [

    PolymerMotif(
        name="aromatic_system",
        priority=50,
        detector="rdkit_ring",
        description="Carbocyclic aromatic ring"
    ),

    PolymerMotif(
        name="heteroaromatic_ring",
        priority=49,
        detector="rdkit_ring",
        description="Aromatic ring containing heteroatoms"
    ),
]


###############################################################################
# Backbone
###############################################################################

BACKBONE_MOTIF = PolymerMotif(

    name="backbone",

    priority=0,

    detector="backbone",

    description="Remaining connected backbone fragment"

)


###############################################################################
# Complete vocabulary
###############################################################################

POLYMER_MOTIFS = sorted(

    FUNCTIONAL_MOTIFS +
    RING_MOTIFS +
    [BACKBONE_MOTIF],

    key=lambda motif: motif.priority,

    reverse=True

)