# IC-wD-MPNN: Interface Conditioned Message Passing for Polymer Property Prediction

IC-wD-MPNN is a motif-aware extension of the weighted directed message-passing neural network (wD-MPNN) for polymer property prediction. It builds on the polymer graph representation introduced by Aldeghi and Coley, incorporating explicit chemical motif information into the message-passing process.

The model retains an atom-level graph representation while introducing a motif representation layer that captures recurring chemical substructures. Motif information is incorporated at polymerization interfaces during message passing, providing an additional level of chemical context for learning polymer representations.

## Model overview

The main components of IC-wD-MPNN are:

- **Polymer-aware graph representation:** Represents polymer inputs through atom-level graphs with fragment and transition weighting information.
- **Motif assignment and pooling:** Identifies chemically meaningful substructures using a predefined SMARTS-based motif vocabulary and aggregates their learned representations.
- **Interface enrichment:** Incorporates pooled motif information into atom representations at polymerization interfaces.
- **Motif-aware message passing:** Allows motif-conditioned information to propagate through the atomic graph for downstream property prediction.

Motifs act as auxiliary representations rather than nodes in a separate hierarchical graph. The underlying computation remains grounded in the original atom-level graph.

For further details, see the [documentation](docs/source/index.rst).

## Installation

The repository includes a Conda environment specification targeting Python 3.10.

```bash
conda env create -f environment.yml
conda activate ic-wd-mpnn
```

The supplied environment provides a baseline CPU-compatible setup. GPU users may need to install a PyTorch build compatible with their CUDA driver and platform.

## Usage

All command-line scripts should be run from the repository root.

### Polymer input format

IC-wD-MPNN uses the polymer string representation introduced in the original wD-MPNN framework. This representation encodes monomer structures, their relative abundances and possible polymerization connections, together with associated transition weights.

For a detailed explanation of the input syntax and worked examples, refer to the original [wD-MPNN repository](https://github.com/coleygroup/polymer-chemprop) and the paper by Aldeghi and Coley.

### Training

Training requires a CSV file containing the polymer input representation and target property columns.

```bash
python train.py \
    --data_path /path/to/train.csv \
    --dataset_type regression \
    --target_columns "target_column" \
    --polymer \
    --save_dir /path/to/checkpoints
```

For separate validation and test datasets, use `--separate_val_path` and `--separate_test_path`.

### Prediction

Predictions can be generated from a trained checkpoint:

```bash
python predict.py \
    --test_path /path/to/test.csv \
    --checkpoint_path /path/to/model.pt \
    --preds_path /path/to/predictions.csv
```

### Fingerprint extraction

Learned representations can be extracted using:

```bash
python fingerprint.py \
    --test_path /path/to/data.csv \
    --checkpoint_path /path/to/model.pt \
    --preds_path /path/to/fingerprints.csv \
    --fingerprint_type MPN
```

The supported fingerprint types are `MPN` and `last_FFN`.

### Hyperparameter optimization

The repository also provides a Hyperopt-based optimization workflow:

```bash
python hyperparameter_optimization.py --help
```

For the complete list of arguments and available options, consult the command-line help:

```bash
python train.py --help
python predict.py --help
python fingerprint.py --help
python hyperparameter_optimization.py --help
```

## Documentation

Detailed documentation covering the polymer representation, model architecture, motif representation, training, prediction, fingerprint extraction and hyperparameter optimization is available in the [`docs/`](docs/source/index.rst) directory.

## Acknowledgements

IC-wD-MPNN builds upon the weighted directed message-passing neural network (wD-MPNN) implementation and polymer representation developed by Aldeghi and Coley.

## Citation

If you use the original polymer wD-MPNN representation, please cite:

```bibtex
@article{wdmpnn,
    title={A graph representation of molecular ensembles for polymer property prediction},
    author={Matteo Aldeghi and Connor W. Coley},
    journal={Chemical Science},
    year={2022},
    volume={13},
    issue={35},
    pages={10486--10498},
    doi={10.1039/D2SC02839E}
}
```

If you use IC-wD-MPNN, please also cite the associated IC-wD-MPNN publication when available.
         publisher="",
         doi="",
         url=""
}
```
