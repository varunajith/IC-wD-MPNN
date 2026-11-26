Command-line reference
======================

The command-line interfaces are defined in :mod:`chemprop.args` and exposed
through scripts in the repository root. The current CLI help output is the
authoritative reference for all options and defaults.

Training
--------

.. code-block:: bash

   python train.py --help

Prediction
----------

.. code-block:: bash

   python predict.py --help

Fingerprint extraction
----------------------

.. code-block:: bash

   python fingerprint.py --help

Hyperparameter optimization
---------------------------

.. code-block:: bash

   python hyperparameter_optimization.py --help

Interpretation
--------------

An interpretation entry point is also present:

.. code-block:: bash

   python interpret.py --help

Its applicability to the polymer workflow should be validated before it is
presented as a supported polymer interpretation method.
