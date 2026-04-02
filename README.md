# FateFormer, a deep model for cell fate prediction

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

## Usage

- **Download data**: Fetch the dataset from https://zenodo.org/17864926 and place all downloaded files inside `datasets/`. Expected files include `clones.csv`, `all_atac_d3_motif.h5ad`, `flux_labelled.csv`, `all_rna_d3_labelled.h5ad` and `all_rna_d3_unlabelled.h5ad`.

- **Train or finetune models (notebooks)**: Each notebook loads data, supports self-supervised pretraining (MLM) and supervised finetuning (CLS) for its modality and evaluation:
  - `Model_RNA.ipynb`
  - `Model_ATAC.ipynb`
  - `Model_Flux.ipynb`
  - `Model_Multimodal.ipynb`

- **Comprehensive analysis script**: `model_analysis.py` runs 4 models (RNA, ATAC, Flux, Multimodal) with 5-fold cross-validation across different seeds (100 runs total). It uses MLM checkpoints from `config.py` and saves outputs under `analysis docs/metrics/` by default:
  - `models/`: trained weights per fold/seed/model
  - `metrics/`: `final_results.csv`, `comprehensive_epoch_results.csv`, `summary_statistics.csv`
  - `fold_results/`: pickled per-fold details
  
  Run with defaults:
  ```bash
  python model_analysis.py
  ```
  Adjust `EPOCHS`, `SELECTION_CRITERIA`, `OUTPUT_FOLDER`, `SEEDS`, or `BATCH_SIZE` by editing the constants in the script.

- **Visualize results**: After `model_analysis.py` finishes, open `Plots.ipynb` to load metrics and artifacts from `analysis docs/metrics/` and generate plots and data summaries.

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



