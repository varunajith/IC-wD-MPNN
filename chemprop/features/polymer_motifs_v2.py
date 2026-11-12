"""Fixed polymer motif vocabulary.

The 112 scaffold motifs are taken directly from the VIPEA Murcko-scaffold
inventory. Assignment is deterministic and uses largest/most-complete
scaffolds before smaller overlapping scaffolds.

Functional-group SMARTS should remain in the existing motif definitions and
be applied only after scaffold assignment.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rdkit import Chem


@dataclass(frozen=True)
class ScaffoldMotif:
    name: str
    smiles: str
    heavy_atoms: int
    ring_count: int
    aromatic_ring_count: int
    fused: bool
    family: str


# ---------------------------------------------------------------------------
# FIXED VOCABULARY: 112 AUDITED MURCKO SCAFFOLDS
# ---------------------------------------------------------------------------

_SCAFFOLD_DATA = [
    {
        'name': 'murcko_001',
        'smiles': 'c1ccc2c(-c3c4ccccc4cc4ccccc34)c3ccccc3cc2c1',
        'heavy_atoms': 28,
        'ring_count': 6,
        'aromatic_ring_count': 6,
        'fused': True,
        'family': 'fused_carbocyclic_aromatic',
    },
    {
        'name': 'murcko_002',
        'smiles': 'c1ccc(NCCCn2c3ccccc3c3ccccc32)cc1',
        'heavy_atoms': 23,
        'ring_count': 4,
        'aromatic_ring_count': 4,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_003',
        'smiles': 'c1ccc2c(-c3cccc4ccccc34)cccc2c1',
        'heavy_atoms': 20,
        'ring_count': 4,
        'aromatic_ring_count': 4,
        'fused': True,
        'family': 'fused_carbocyclic_aromatic',
    },
    {
        'name': 'murcko_004',
        'smiles': 'c1ccc(-n2c3ccccc3c3ccccc32)cc1',
        'heavy_atoms': 19,
        'ring_count': 4,
        'aromatic_ring_count': 4,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_005',
        'smiles': 'c1ccc(N(c2ccccc2)c2ccccc2)cc1',
        'heavy_atoms': 19,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_006',
        'smiles': 'c1ccc2c(c1)c1ccccc1c1ccccc21',
        'heavy_atoms': 18,
        'ring_count': 4,
        'aromatic_ring_count': 4,
        'fused': True,
        'family': 'fused_carbocyclic_aromatic',
    },
    {
        'name': 'murcko_007',
        'smiles': 'c1ccc2c(c1)-c1ccccc1C21CCOCC1',
        'heavy_atoms': 18,
        'ring_count': 4,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_008',
        'smiles': 'c1ccc(-c2ccc(-c3ccccc3)cc2)cc1',
        'heavy_atoms': 18,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': False,
        'family': 'carbocyclic_aromatic',
    },
    {
        'name': 'murcko_009',
        'smiles': 'c1ccc2c(c1)c1ccccc1n2CC1CO1',
        'heavy_atoms': 17,
        'ring_count': 4,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_010',
        'smiles': 'c1cc2ccc3cccc4ccc(c1)c2c34',
        'heavy_atoms': 16,
        'ring_count': 4,
        'aromatic_ring_count': 4,
        'fused': True,
        'family': 'fused_carbocyclic_aromatic',
    },
    {
        'name': 'murcko_011',
        'smiles': 'O=c1oc(=O)c2cc3c(=O)oc(=O)c3cc12',
        'heavy_atoms': 16,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_012',
        'smiles': 'c1ccc(-c2ccc3ccccc3n2)cc1',
        'heavy_atoms': 16,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_013',
        'smiles': 'c1ccc(-c2cnc3ncncc3c2)cc1',
        'heavy_atoms': 16,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_014',
        'smiles': 'O=C1C(=O)c2ccccc2-c2ccccc21',
        'heavy_atoms': 16,
        'ring_count': 3,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_015',
        'smiles': 'O=C1c2ccccc2C(=O)c2ccccc21',
        'heavy_atoms': 16,
        'ring_count': 3,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_016',
        'smiles': 'c1ccc(-c2nc3ccccc3s2)cc1',
        'heavy_atoms': 15,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_017',
        'smiles': 'c1csc(-c2ccc(-c3cccs3)s2)c1',
        'heavy_atoms': 15,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_018',
        'smiles': 'O=S1(=O)c2ccccc2-c2ccccc21',
        'heavy_atoms': 15,
        'ring_count': 3,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_019',
        'smiles': 'O=C(Nc1ccccc1)c1ccccc1',
        'heavy_atoms': 15,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_020',
        'smiles': 'S=C(Nc1ccccc1)c1ccccc1',
        'heavy_atoms': 15,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_021',
        'smiles': 'c1ccc2cc3ccccc3cc2c1',
        'heavy_atoms': 14,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_carbocyclic_aromatic',
    },
    {
        'name': 'murcko_022',
        'smiles': 'c1cnc2c(c1)ccc1cccnc12',
        'heavy_atoms': 14,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_023',
        'smiles': 'C=C1c2ccccc2-c2ccccc21',
        'heavy_atoms': 14,
        'ring_count': 3,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_024',
        'smiles': 'O=C1c2ccccc2-c2ccccc21',
        'heavy_atoms': 14,
        'ring_count': 3,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_025',
        'smiles': 'O=C(c1ccccc1)c1ccccc1',
        'heavy_atoms': 14,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_026',
        'smiles': 'c1ccc(COc2ccccc2)cc1',
        'heavy_atoms': 14,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_027',
        'smiles': 'c1ccc(COc2ccccn2)cc1',
        'heavy_atoms': 14,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_028',
        'smiles': 'c1ccc(CNC2CCCCC2)cc1',
        'heavy_atoms': 14,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'mixed_aromatic',
    },
    {
        'name': 'murcko_029',
        'smiles': 'c1ccc(OCC2CCCCO2)cc1',
        'heavy_atoms': 14,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'mixed_aromatic',
    },
    {
        'name': 'murcko_030',
        'smiles': 'c1ccc2c(c1)[nH]c1ccccc12',
        'heavy_atoms': 13,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_031',
        'smiles': 'c1ccc2c(c1)oc1ccccc12',
        'heavy_atoms': 13,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_032',
        'smiles': 'c1ccc2c(c1)sc1ccccc12',
        'heavy_atoms': 13,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_033',
        'smiles': 'c1ccc2cc3nsnc3cc2c1',
        'heavy_atoms': 13,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_034',
        'smiles': 'c1ccc2c(c1)Cc1ccccc1-2',
        'heavy_atoms': 13,
        'ring_count': 3,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_035',
        'smiles': 'c1ccc(Oc2ccccc2)cc1',
        'heavy_atoms': 13,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_036',
        'smiles': 'c1cc2cc3sccc3cc2s1',
        'heavy_atoms': 12,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_037',
        'smiles': 'O=C1c2ccsc2-c2sccc21',
        'heavy_atoms': 12,
        'ring_count': 3,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_038',
        'smiles': 'c1c2c(cc3nsnc13)N=S=N2',
        'heavy_atoms': 12,
        'ring_count': 3,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_039',
        'smiles': 'c1ccc(-c2ccccc2)cc1',
        'heavy_atoms': 12,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': False,
        'family': 'carbocyclic_aromatic',
    },
    {
        'name': 'murcko_040',
        'smiles': 'c1ccc(-c2ccccn2)nc1',
        'heavy_atoms': 12,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_041',
        'smiles': 'c1ccc(Cn2cncn2)cc1',
        'heavy_atoms': 12,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_042',
        'smiles': 'c1ccc(N2CCCCC2)cc1',
        'heavy_atoms': 12,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'mixed_aromatic',
    },
    {
        'name': 'murcko_043',
        'smiles': 'c1ccc(N2CCOCC2)cc1',
        'heavy_atoms': 12,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'mixed_aromatic',
    },
    {
        'name': 'murcko_044',
        'smiles': 'c1cc2sc3ccsc3c2s1',
        'heavy_atoms': 11,
        'ring_count': 3,
        'aromatic_ring_count': 3,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_045',
        'smiles': 'O=C1CCOc2ccccc21',
        'heavy_atoms': 11,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_046',
        'smiles': 'O=C1CCc2ccccc2N1',
        'heavy_atoms': 11,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_047',
        'smiles': 'O=C1Nc2ccccc2C1=O',
        'heavy_atoms': 11,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_048',
        'smiles': 'O=C1OC(=O)c2ccccc21',
        'heavy_atoms': 11,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_049',
        'smiles': 'c1ccc(N2CCCC2)cc1',
        'heavy_atoms': 11,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'mixed_aromatic',
    },
    {
        'name': 'murcko_050',
        'smiles': 'c1ncn(C2CCCCO2)n1',
        'heavy_atoms': 11,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'mixed_aromatic',
    },
    {
        'name': 'murcko_051',
        'smiles': 'O=c1[nH][nH]c2ccccc12',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_052',
        'smiles': 'c1ccc2ccccc2c1',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_carbocyclic_aromatic',
    },
    {
        'name': 'murcko_053',
        'smiles': 'c1ccc2cnccc2c1',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_054',
        'smiles': 'c1ccc2ncccc2c1',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_055',
        'smiles': 'c1ccc2nccnc2c1',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_056',
        'smiles': 'c1ccc2ncncc2c1',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_057',
        'smiles': 'c1cnc2cccnc2c1',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_058',
        'smiles': 'c1cnc2cnccc2c1',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_059',
        'smiles': 'c1cnc2ncccc2c1',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_060',
        'smiles': 'c1csc(-c2cccs2)c1',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_061',
        'smiles': 'O=C1Cc2ccccc2N1',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_062',
        'smiles': 'O=C1OC(=O)c2cscc21',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_063',
        'smiles': 'c1ccc2c(c1)CCCN2',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_064',
        'smiles': 'c1ccc2c(c1)CCCO2',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_065',
        'smiles': 'c1cnc2c(c1)NCCC2',
        'heavy_atoms': 10,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_066',
        'smiles': 'c1cc2[nH]ccc2cn1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_067',
        'smiles': 'c1cc2c[nH]nc2cn1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_068',
        'smiles': 'c1cc2n[nH]cc2cn1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_069',
        'smiles': 'c1cc2ncsc2cn1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_070',
        'smiles': 'c1cc2nsnc2cn1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_071',
        'smiles': 'c1ccc2[nH]ccc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_072',
        'smiles': 'c1ccc2[nH]cnc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_073',
        'smiles': 'c1ccc2[nH]ncc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_074',
        'smiles': 'c1ccc2n[nH]cc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_075',
        'smiles': 'c1ccc2nonc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_076',
        'smiles': 'c1ccc2nsnc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_077',
        'smiles': 'c1ccc2occc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_078',
        'smiles': 'c1ccc2scnc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_079',
        'smiles': 'c1ccn2ccnc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_080',
        'smiles': 'c1ccn2cncc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_081',
        'smiles': 'c1ccn2ncnc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_082',
        'smiles': 'c1cn2ccnc2cn1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_083',
        'smiles': 'c1cn2ncnc2cn1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_084',
        'smiles': 'c1cnc2[nH]ccc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_085',
        'smiles': 'c1cnc2[nH]ccc2n1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_086',
        'smiles': 'c1cnc2ccnn2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_087',
        'smiles': 'c1cnc2n[nH]cc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_088',
        'smiles': 'c1cnn2ccnc2c1',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_089',
        'smiles': 'c1ccc2c(c1)CCO2',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_090',
        'smiles': 'c1ccc2c(c1)OCO2',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_091',
        'smiles': 'c1scc2c1OCCO2',
        'heavy_atoms': 9,
        'ring_count': 2,
        'aromatic_ring_count': 1,
        'fused': True,
        'family': 'fused_mixed_aromatic',
    },
    {
        'name': 'murcko_092',
        'smiles': 'c1cc2cscc2s1',
        'heavy_atoms': 8,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_093',
        'smiles': 'c1cc2sccc2s1',
        'heavy_atoms': 8,
        'ring_count': 2,
        'aromatic_ring_count': 2,
        'fused': True,
        'family': 'fused_heteroaromatic',
    },
    {
        'name': 'murcko_094',
        'smiles': 'O=C1C=CC(=O)C=C1',
        'heavy_atoms': 8,
        'ring_count': 1,
        'aromatic_ring_count': 0,
        'fused': False,
        'family': 'non_aromatic_cyclic',
    },
    {
        'name': 'murcko_095',
        'smiles': 'O=c1cc[nH]cc1',
        'heavy_atoms': 7,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_096',
        'smiles': 'O=c1cccc[nH]1',
        'heavy_atoms': 7,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_097',
        'smiles': 'O=c1cncc[nH]1',
        'heavy_atoms': 7,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_098',
        'smiles': 'O=S1(=O)C=CC=C1',
        'heavy_atoms': 7,
        'ring_count': 1,
        'aromatic_ring_count': 0,
        'fused': False,
        'family': 'non_aromatic_cyclic',
    },
    {
        'name': 'murcko_099',
        'smiles': 'c1cc[nH+]cc1',
        'heavy_atoms': 6,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_100',
        'smiles': 'c1ccccc1',
        'heavy_atoms': 6,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'carbocyclic_aromatic',
    },
    {
        'name': 'murcko_101',
        'smiles': 'c1ccncc1',
        'heavy_atoms': 6,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_102',
        'smiles': 'c1ccnnc1',
        'heavy_atoms': 6,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_103',
        'smiles': 'c1cnccn1',
        'heavy_atoms': 6,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_104',
        'smiles': 'c1cncnc1',
        'heavy_atoms': 6,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_105',
        'smiles': 'c1c[nH]cn1',
        'heavy_atoms': 5,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_106',
        'smiles': 'c1cc[nH]c1',
        'heavy_atoms': 5,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_107',
        'smiles': 'c1ccoc1',
        'heavy_atoms': 5,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_108',
        'smiles': 'c1ccsc1',
        'heavy_atoms': 5,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_109',
        'smiles': 'c1cn[nH]c1',
        'heavy_atoms': 5,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_110',
        'smiles': 'c1cscn1',
        'heavy_atoms': 5,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_111',
        'smiles': 'c1nc[nH]n1',
        'heavy_atoms': 5,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
    {
        'name': 'murcko_112',
        'smiles': 'c1nncs1',
        'heavy_atoms': 5,
        'ring_count': 1,
        'aromatic_ring_count': 1,
        'fused': False,
        'family': 'heteroaromatic',
    },
]


def _build_vocabulary() -> Tuple[ScaffoldMotif, ...]:
    motifs = []

    for data in _SCAFFOLD_DATA:
        mol = Chem.MolFromSmiles(data["smiles"])
        if mol is None:
            raise ValueError(f"Invalid scaffold SMILES: {data['smiles']}")

        # Verify the inventory against RDKit.
        if mol.GetNumHeavyAtoms() != data["heavy_atoms"]:
            raise ValueError(
                f"Heavy-atom mismatch for {data['smiles']}: "
                f"inventory={data['heavy_atoms']}, "
                f"RDKit={mol.GetNumHeavyAtoms()}"
            )

        motifs.append(ScaffoldMotif(**data))

    return tuple(motifs)


# Already ordered largest -> smallest.
_FULL_SCAFFOLD_MOTIFS = _build_vocabulary()

assert len(_FULL_SCAFFOLD_MOTIFS) == 112
assert len({m.smiles for m in _FULL_SCAFFOLD_MOTIFS}) == 112




# ---------------------------------------------------------------------------
# FUNCTIONAL-GROUP VOCABULARY
# ---------------------------------------------------------------------------
#
# These motifs are applied AFTER scaffold assignment. Their priority only
# resolves overlaps between functional-group matches; it does not compete
# with the 112 scaffold motifs.
#
# A functional-group match may contain atoms that already belong to a
# scaffold. The assignment code in featurization.py must therefore exclude
# already-claimed atoms when creating functional-group instances.

@dataclass(frozen=True)
class FunctionalMotif:
    name: str
    priority: int
    smarts: str
    description: str = ""
    examples: Optional[List[str]] = None


FUNCTIONAL_MOTIFS = [
    # Carbonyl chemistry: most specific first.
    FunctionalMotif(
        name="ester",
        priority=100,
        smarts="[#6](=O)-[#8]-[#6]",
        description="Ester linkage",
        examples=["PET", "PMMA"],
    ),
    FunctionalMotif(
            name="imide",
            priority=96,
            smarts="[CX3](=O)[NX3][CX3](=O)",
            description="Imide group",
            examples=["Polyimide"],
        ),
    FunctionalMotif(
        name="amide",
        priority=95,
        smarts="[#6](=O)-[#7]",
        description="Amide linkage",
        examples=["Nylon"],
    ),

    FunctionalMotif(
        name="carboxylic_acid",
        priority=93,
        smarts="[CX3](=O)[OX2H1]",
        description="Carboxylic acid",
    ),
    FunctionalMotif(
        name="acid_chloride",
        priority=92,
        smarts="[CX3](=O)[Cl]",
        description="Acid chloride",
    ),
    FunctionalMotif(
        name="carbonyl",
        priority=90,
        smarts="[CX3](=O)",
        description="Remaining carbonyl group",
    ),

    # Oxygen chemistry.
    FunctionalMotif(
        name="ether",
        priority=85,
        smarts="[OX2]([#6])[#6]",
        description="Ether linkage",
        examples=["PEEK"],
    ),
    FunctionalMotif(
        name="hydroxyl",
        priority=80,
        smarts="[OX2H1]",
        description="Alcohol hydroxyl group",
    ),
    FunctionalMotif(
            name="nitro",
            priority=76,
            smarts="[N+](=O)[O-]",
            description="Nitro group",
    ),

    # Nitrogen chemistry.
    FunctionalMotif(
        name="amine",
        priority=75,
        smarts="[NX3;!$(N-C(=O));!$(N-S(=O)(=O))]",
        description="Amine nitrogen",
    ),
    FunctionalMotif(
        name="nitrile",
        priority=74,
        smarts="[#6]#[#7]",
        description="Nitrile group",
        examples=["PAN"],
    ),


    # Sulfur chemistry.
    FunctionalMotif(
        name="sulfone",
        priority=70,
        smarts="[SX4](=O)(=O)([#6])[#6]",
        description="Sulfone group",
    ),
    FunctionalMotif(
        name="sulfoxide",
        priority=69,
        smarts="[SX3](=O)([#6])[#6]",
        description="Sulfoxide group",
    ),
    FunctionalMotif(
        name="thioether",
        priority=68,
        smarts="[SX2]([#6])[#6]",
        description="Thioether linkage",
    ),

    FunctionalMotif(
    name="trifluoromethyl",
    priority=61,
    smarts="[#6X4]([#9])([#9])[#9]",
    description="Trifluoromethyl group (CF3)",
   ) ,

    # Halogens are retained as functional annotations. They are deliberately
    # below the chemically larger scaffold motifs.
    FunctionalMotif(
    name="fluoro",
    priority=60,
    smarts="[#9]",
    description="Fluorine substituent",
    ),

    FunctionalMotif(
    name="chloro",
    priority=59,
    smarts="[#17]",
    description="Chlorine substituent",
    ),

    FunctionalMotif(
    name="bromo",
    priority=58,
    smarts="[#35]",
    description="Bromine substituent",
    ),

    FunctionalMotif(
    name="iodo",
    priority=57,
    smarts="[#53]",
    description="Iodine substituent",
    ),

]
# ---------------------------------------------------------------------------
# BACKBONE FALLBACK
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackboneMotif:
    name: str = "backbone"
    priority: int = 0
    detector: str = "backbone"
    description: str = "Remaining connected atoms after scaffold and functional-group assignment"


BACKBONE_MOTIF = BackboneMotif()

FUNCTIONAL_MOTIFS.sort(key=lambda x: x.priority, reverse=True)


# ---------------------------------------------------------------------------
# COMPLETE MOTIF VOCABULARY
# ---------------------------------------------------------------------------

SCAFFOLD_MOTIFS = _FULL_SCAFFOLD_MOTIFS
FUNCTIONAL_MOTIFS = tuple(FUNCTIONAL_MOTIFS)



__all__ = [
    "ScaffoldMotif",
    "FunctionalMotif",
    "BackboneMotif",
    "SCAFFOLD_MOTIFS",
    "FUNCTIONAL_MOTIFS",
    "BACKBONE_MOTIF",
]
