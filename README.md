## Overview

ScalarGNNs introduces a few novel models based off the Graph Convolutional Network (GCN), Simple Graph Convolution (SGC), and Graph Attention Network (GAT) architectures.

Main Models:
- ScalarGCN
- ScalarSGC
- ScalarSAFGATv2

## Setup
This implementation is based on Pytorch. We assume that you're using Python 3 with pip installed. To run the code, you need the following dependencies:

- [Pytorch 2.1](https://pytorch.org/)
- [Pytorch Geometric 2.6.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

We utilise pytorch geometric to load data. However. you can upload any graph datasets as you want, and change the load_data function in /utils.py.

## Files
- main.py: The main script used for executing experiments
- samplers.py: Sampling methods (full-restricted, GraphSAGE, FastGCN, LADIES)
- models.py: The GNN models
- utils.py: Utility functions
- plot_loss.py: An example file that can be used to visualise the per_run data generated with the log_runs flag set to True.

## Experiment Hyperaparameters
The experiment hyperparameters should all be pretty self-explanatory and more information abuout them can be shown by running 

```bash
python3 main.py --h # Displays information about hyperaparameters
```

The argparse.BooleanOptionalAction hyperparameters are by default set to false.

Some hyperparameters that require further explanation:
- log_memory_snapshot: This generates a memory snapshot entitled "mem_log.pickle" in the main folder. This snapshot can then be uploaded to the [Pytorch Snapshot Visualisation Tool](https://pytorch.org/memory_viz) to visualise how memory is allocated across the run.
- log_runs: This logs information to a different file for each run in results/per_epoch. The file is named with the main hyperparameters for the run, as well as the run number. The epoch number, train loss, validation loss, and validation f1 are logged in each line of the file.
- log_final: This writes all the final information from a set of runs to the file "results/final.csv". The comma separated values represent in the follow order: "pretrain_mem, pretrain_time, total_time, model_size, total_params, train_params, train_time, valid_time, epochs, best_epoch, train_time_per_epoch, train_memory, val_memory, adjs_memory, val_acc, val_f1, val_sens, val_spec, test_acc, test_f1, test_sens, test_spec".

## Usage
Example of command line usage:

```bash
python3 main.py --dataset cora --model scalarGCN --sampler LADIES  # Train ScalarGCN with LADIES on cora.
```

Modify the hyperparameters as required.