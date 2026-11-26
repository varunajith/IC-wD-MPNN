Installation
============

Environment
-----------

The repository provides an environment specification in
:file:`environment.yml`. From the repository root, create and activate it:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate ic-wd-mpnn

If the environment already exists:

.. code-block:: bash

   conda env update -f environment.yml --prune

The environment specification targets Python 3.10 and provides a baseline CPU-compatible setup with the core dependencies used by the implementation, including PyTorch, RDKit, NumPy, pandas, scikit-learn, Hyperopt, TensorBoardX, tqdm and typed-argument-parser.

Running from source
-------------------

Run the command-line scripts from the repository root:

.. code-block:: bash

   python train.py --help
   python predict.py --help
   python fingerprint.py --help
   python hyperparameter_optimization.py --help

This repository is intended to be used from its source checkout. The
inherited PyPI, editable-installation and Docker instructions do not apply
to the current release.

Hardware
--------

The supplied environment is a baseline environment. GPU users may need to
install a PyTorch build compatible with their CUDA driver and platform;
see the official `PyTorch installation instructions
<https://pytorch.org/get-started/locally/>`_. Use ``--no_cuda`` to force
CPU execution.
