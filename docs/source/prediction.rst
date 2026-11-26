Prediction
==========

Use :file:`predict.py` to generate predictions from a trained checkpoint.

Basic prediction
----------------

.. code-block:: bash

   python predict.py \
       --test_path /path/to/test.csv \
       --checkpoint_path /path/to/model.pt \
       --preds_path /path/to/predictions.csv \
       --num_workers 2

The input file must contain the polymer-input column expected by the
checkpoint configuration. The output predictions are written to the
specified CSV path.

Ensembles and options
---------------------

The prediction interface supports checkpoint paths and checkpoint
directories according to the CLI options. Inspect the current help output
for the exact syntax:

.. code-block:: bash

   python predict.py --help

CPU execution
-------------

Add ``--no_cuda`` to force CPU execution. Use ``--num_workers 0`` to avoid
DataLoader multiprocessing.

Only load trusted checkpoint files. PyTorch checkpoint loading uses Python
deserialisation; loading an untrusted checkpoint can execute arbitrary code.
