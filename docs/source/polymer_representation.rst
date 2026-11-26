Polymer input representation
============================

IC-wD-MPNN uses polymer-aware molecule construction and featurisation.
Polymer inputs can encode multiple molecular fragments and associated
composition or weighting information; they should not be treated as
ordinary single-molecule SMILES unless the dataset preparation procedure
specifies that conversion.

Input data
----------

Training and prediction data are supplied as CSV files with a header row.
The polymer input column is selected using ``--smiles_columns`` for
prediction/fingerprint workflows, while target properties are selected
using ``--target_columns`` during training. Use the exact column names
provided in the CSV header.

Polymer graph construction
--------------------------

The polymer-aware molecule construction code is implemented in
:mod:`chemprop.rdkit`. The polymer path combines component fragments into a
molecular graph and retains fragment-weight information on atoms. The
featurisation pipeline recognises polymer inputs and handles the
polymer-specific information when constructing graph objects.

The underlying graph remains atom-based: atoms are graph nodes and directed
bonds are used for message passing. The motif representation supplements
this graph; it does not replace it with a separate hierarchical graph.

Preparing splits
----------------

The repository includes :file:`prepare-cv-splits.py` for preparing
train/validation/test CSV files from a precomputed cross-validation split
file. Inspect its current arguments and expected inputs with:

.. code-block:: bash

   python prepare-cv-splits.py --help

When supplying separate validation and test sets, use
``--separate_val_path`` and ``--separate_test_path`` during training.
Maintain consistent input and target column names across the files.
