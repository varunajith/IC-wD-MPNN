Latent fingerprint extraction
==============================

Use :file:`fingerprint.py` to export learned representations from a trained
IC-wD-MPNN checkpoint. These vectors can support downstream analysis,
visualisation or other modelling workflows.

Fingerprint types
-----------------

``MPN``
   Returns the representation produced by the message-passing encoder.
   In this implementation, this representation includes the motif-aware
   polymer message-passing computation.

``last_FFN``
   Returns the feed-forward network representation immediately before
   the final prediction layer.

Extraction
----------

.. code-block:: bash

   python fingerprint.py \
       --test_path /path/to/data.csv \
       --checkpoint_path /path/to/model.pt \
       --preds_path /path/to/fingerprints.csv \
       --fingerprint_type MPN \
       --smiles_columns polymer_input \
       --num_workers 2

Replace ``polymer_input`` with the actual polymer-input column name.
Select ``last_FFN`` to extract the alternative representation.

Inspect the complete option list with:

.. code-block:: bash

   python fingerprint.py --help

Output
------

The output CSV contains the input polymer column and one column per
fingerprint dimension (for example, ``fp_0``, ``fp_1`` and so on).
The dimensionality depends on the model configuration and fingerprint
type.
