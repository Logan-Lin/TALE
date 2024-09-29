# TALE: Time-Aware Location Embedding

This repository provides the implementation code for the Time-Aware Location Embedding (TALE) model, as proposed in the TKDE paper: *Pre-Training Time-Aware Location Embeddings from Spatial-Temporal Trajectories*.

## Model overview

![model structure](figure/tale_model.png)

TALE is a location embedding model which maps each location (typically a POI) to a low dimensional embedding space. It utilizes trajectory data (or check-in data) to extract location functionalities from contextual and temporal information. In addition to the contextual information extracted by conventional word2vec models, TALE introduces a temporal tree structure that segments time into intervals and utilizes hierarchical softmax to further incorporate temporal information.

## Code overview

This repository contains various types of files:

- `run.py`: Entry point for training and evaluating TALE and other baselines.
- `dataset.py`: Dataset classes for loading and preprocessing trajectory data.
- `utils.py`: Utility functions for data loading, evaluation, and other tasks.
- `datasets` directory: Contains sample files for the datasets.
- `embed` directory: Contains the implementation of TALE and other embedding models.
- `downstream` directory: Contains the implementation of downstream tasks and models.

## Requirements

- python >= 3.7
- pytorch == 1.6.0
- scikit-learn
- numpy == 1.19.1
- pandas == 1.1.2
- tables

## Paper information

Reference: 

> Huaiyu Wan, Yan Lin, Shengnan Guo, Youfang Lin. "Pre-training time-aware location embeddings from spatial-temporal trajectories." IEEE Transactions on Knowledge and Data Engineering 34.11 (2021): 5510-5523.

Paper: https://ieeexplore.ieee.org/document/9351627

If you have any further questions, feel free to contact me directly. My contact information is available on my homepage: https://www.yanlincs.com/