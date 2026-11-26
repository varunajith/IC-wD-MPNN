Training
========

Data
----

Training data are supplied as a CSV file with a header row. The file must
contain the polymer input column and one or more target columns. Use
``--target_columns`` to identify the property or properties to predict
and ``--polymer`` to enable the polymer-specific model path.

Basic training
--------------

Run from the repository root:

.. code-block:: bash

   python train.py \
    --data_path /path/to/train.csv \
    --smiles_columns "polymer_input" \
    --dataset_type regression \
    --target_columns "target_column" \
    --polymer \
    --save_dir /path/to/checkpoints

The target-column name must match the CSV header. Quote names containing
spaces or special characters.

Separate validation and test sets
---------------------------------

Provide independent validation and test files with
``--separate_val_path`` and ``--separate_test_path``:

.. code-block:: bash

   python train.py \
       --data_path /path/to/train.csv \
       --separate_val_path /path/to/validation.csv \
       --separate_test_path /path/to/test.csv \
       --smiles_columns "polymer_input" \
       --dataset_type regression \
       --target_columns "target_column" \
       --polymer \
       --save_dir /path/to/checkpoints

Keep input and target column names consistent across the files.

Cross-validation and predefined monomer splits
-----------------------------------------------

The repository includes `prepare-cv-splits.py` to prepare the data files used in the nine-fold monomer-split evaluation on the VIPEA dataset. Given the dataset CSV and the precomputed split indices, the script generates the corresponding training, validation and test CSV files for each of the nine folds.

This utility is specific to the predefined VIPEA monomer-split comparison used in the IC-wD-MPNN experiments. It prepares the input files; it does not implement the cross-validation procedure itself.

Cross-validation is handled separately by the model's existing training and data-splitting functionality. The available split modes and cross-validation arguments can be inspected using:

.. code-block:: bash

   python train.py --help

To inspect the arguments and expected inputs of the VIPEA split-preparation utility, run:

.. code-block:: bash

   python prepare-cv-splits.py --help

Training options
----------------

The CLI exposes dataset, model, optimisation, validation, checkpoint and
device options. The checked-out source is authoritative; inspect the
current options with:

.. code-block:: bash

   python train.py --help

Use ``--no_cuda`` for CPU execution and ``--num_workers 0`` to disable
DataLoader worker processes.

Reproducibility
---------------

Record the random seed, data split, checkpoint configuration and software
environment for each experiment. Keep the final test set out of model
selection and hyperparameter optimisation.
