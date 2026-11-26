Hyperparameter optimization
===========================

The repository includes a Hyperopt-based optimisation workflow. Its
implementation is in :mod:`chemprop.hyperparameter_optimization`, with
:file:`hyperparameter_optimization.py` as the command-line entry point.

The current search space covers hidden size, message-passing depth,
dropout and the number of feed-forward layers. The workflow accepts the
``--polymer`` option. It does not automatically optimise every
motif-specific setting.

Inspect the available options
-----------------------------

.. code-block:: bash

   python hyperparameter_optimization.py --help

Example invocation
------------------

.. code-block:: bash

   python hyperparameter_optimization.py \
       --data_path /path/to/train.csv \
       --dataset_type regression \
       --target_columns "target_column" \
       --polymer \
       --num_iters 20 \
       --config_save_path /path/to/best_config.json

Use the exact argument names and requirements shown by the current
``--help`` output.

Evaluation discipline
---------------------

Do not use the final held-out test set for hyperparameter selection.
Optimisation should use only the training and validation data available
within the experimental split. For cross-validation experiments, optimise
within each split when that matches the evaluation protocol.
