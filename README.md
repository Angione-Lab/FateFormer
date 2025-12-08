# Cell Fate Prediction

A deep learning framework for predicting cell fate using multimodal single-cell data (RNA-seq, ATAC-seq, and metabolic flux). This project employs transformer-based architectures with masked language modeling (MLM) pretraining to predict whether cells will undergo reprogramming or reach a dead-end state.

## Overview

This repository implements a multimodal transformer architecture that integrates:
- **RNA-seq data**: Gene expression profiles
- **ATAC-seq data**: Chromatin accessibility measurements  
- **Metabolic flux data**: Metabolic reaction fluxes

The framework supports both single-modality and multimodal prediction tasks, with optional MLM pretraining for improved performance.

## Features

- **Single-modality models**: Transformer-based models for RNA, ATAC, or Flux data individually
- **Multimodal integration**: Combines all three modalities using attention mechanisms
- **MLM pretraining**: Optional masked language modeling pretraining for better feature learning
- **Cross-validation**: Stratified k-fold cross-validation with comprehensive metrics
- **Interpretability tools**: Attention visualization, latent space analysis, and feature importance
- **Flexible data handling**: Supports missing modalities in multimodal scenarios

## Project Structure

```
CellFatePrediction/
├── data/                    # Data loading and preprocessing modules
│   ├── load_data.py        # Functions to load RNA, ATAC, and Flux data
│   ├── preprocess_data.py  # Data preprocessing and filtering
│   └── create_dataset.py   # Dataset creation and pairing utilities
├── models/                  # Model architectures
│   └── transformers.py     # SingleTransformer and MultiModalTransformer
├── utils/                   # Utility functions
│   ├── helpers.py          # Helper functions for model creation and data handling
│   └── losses.py           # Custom loss functions (MLM loss, etc.)
├── interpretation/          # Model interpretation tools
│   ├── attentions.py       # Attention weight analysis
│   ├── latentspace.py      # Latent space visualization
│   ├── predictions.py      # Prediction analysis
│   ├── similarity.py       # Similarity metrics
│   └── visualization.py    # Visualization utilities
├── train.py                # Training functions (MLM and classification)
├── evaluate.py             # Evaluation functions
├── config.py               # Configuration file (checkpoint paths, seed)
├── datasets/               # Data files (pickle, CSV)
├── objects/                # Saved objects (DEGs, feature importance, etc.)
└── *.ipynb                 # Jupyter notebooks for model training and analysis
```

## Installation

### Requirements

The project requires Python 3.7+ and the following packages:

```python
torch>=1.9.0
numpy
pandas
scikit-learn
anndata
scanpy
tqdm
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CellFatePrediction
```

2. Install dependencies:
```bash
pip install torch numpy pandas scikit-learn anndata scanpy tqdm
```

## Data Format

### Input Data

The framework expects data in the following formats:

- **RNA data**: AnnData object (`.h5ad`) with gene expression counts
- **ATAC data**: AnnData object (`.h5ad`) with chromatin accessibility scores
- **Flux data**: CSV file with metabolic reaction fluxes (rows = cells, columns = reactions)
- **Clone information**: CSV file with clone IDs and cell fate labels

### Required Metadata

Each dataset should include:
- `clone_id`: Clone identifier for pairing cells across modalities
- `batch_no`: Batch information (0, 1, or 2)
- `label`: Cell fate label ('reprogramming' or 'dead-end')

## Model Architecture

### SingleTransformer

Each modality uses a transformer encoder with:
- **Token embeddings**: Count-based embeddings for RNA, linear projection for ATAC/Flux
- **Position embeddings**: Learnable positional encodings
- **Batch embeddings**: Batch effect correction
- **Transformer encoder**: Multi-head self-attention layers
- **CLS token**: Classification token with cross-attention to all tokens

### MultiModalTransformer

The multimodal model:
1. Processes each modality independently using SingleTransformer
2. Concatenates token representations from all modalities
3. Uses a CLS token with attention to all modality tokens
4. Applies masking to handle missing modalities

## Training Details

### Loss Functions

- **MLM Loss**: Masked language modeling loss (MSE or cross-entropy)
- **Classification Loss**: 
  - Binary cross-entropy (BCE)
  - Weighted BCE (handles class imbalance)
  - Focal loss (handles hard examples)

### Evaluation Metrics

- AUC-ROC
- Precision, Recall, F1-score
- Accuracy
- Specificity

The framework reports metrics for:
- All validation samples
- Common samples (samples with all modalities)
- Modality-specific subsets (e.g., RNA+Flux, ATAC-only)

## Configuration

Edit `config.py` to set:
- MLM checkpoint paths
- Random seed
- Other global parameters

## Notebooks

The repository includes Jupyter notebooks for:
- `Model_RNA.ipynb`: RNA-only model training
- `Model_ATAC.ipynb`: ATAC-only model training
- `Model_Flux.ipynb`: Flux-only model training
- `Model_Multimodal.ipynb`: Multimodal model training
- `Plots.ipynb`: Visualization and analysis



