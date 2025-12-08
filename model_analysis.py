#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Cell Fate Prediction Models

This script runs a comprehensive analysis of 4 models (Multimodal, RNA, ATAC, Flux)
with 5-fold cross-validation and 5 different seeds (125 runs total).

For each run, it evaluates on both all samples and common samples (163 samples),
saves metrics from the best epoch and stores results in CSV format.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from utils.helpers import get_max
from train import weighted_bce_loss
from data import load_data, create_dataset


def setup_analysis_folder(output_folder="analysis_results"):
    """Create analysis results folder structure."""
    script_dir = Path(__file__).parent
    results_dir = os.path.join(script_dir, output_folder)
    
    # Create main results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ["models", "metrics", "fold_results"]
    for subdir in subdirs:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
    
    return results_dir

def load_all_data():
    """Load all required datasets."""
    print("Loading data...")
    
    # Load RNA data
    try:
        with open('objects/rna_labelled.pkl', 'rb') as f:
            adata_RNA_labelled = pickle.load(f)
        with open('objects/rna_unlabelled.pkl', 'rb') as f:
            adata_RNA_unlabelled = pickle.load(f)
        with open('objects/degs.pkl', 'rb') as f:
            df_degs = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading RNA data: {e}")
        raise
    rna_vocab_size = int(get_max([adata_RNA_labelled, adata_RNA_unlabelled])) + 2
    del adata_RNA_unlabelled
    
    # Convert RNA AnnData to PyTorch dataset for single modality training
    rna_dataset, _, _ = create_dataset.get_cls_dataset(
        data=adata_RNA_labelled,
        batch_key="batch_no",
        label_key="label", 
        pct_key="pct",
        filter_pcts=0.0,
        data_dtype=torch.int32
    )
    # Load ATAC data
    try:
        adata_ATAC_labelled, _ = load_data.load_atac(
            data_path="data/datasets/atac/all_atac_d3_motif.h5ad",
            clone_info=True, 
            clone_path="data/datasets/clone/clones.csv"
        )
        
        # Convert ATAC AnnData to PyTorch dataset for single modality training
        atac_dataset, _, _ = create_dataset.get_cls_dataset(
            data=adata_ATAC_labelled,
            batch_key="batch_no",
            label_key="label",
            pct_key="pct", 
            filter_pcts=0.0,
            data_dtype=torch.float32
        )
    except FileNotFoundError as e:
        print(f"Error loading ATAC data: {e}")
        raise
    
    # Load Flux data
    try:
        fluxes = load_data.load_flux("data/datasets/flux/flux_labelled_11nov.csv", prefix="flux_un",
                                                                        clone_info=True, 
                                                                        clone_path="data/datasets/clone/clones.csv", 
                                                                        scale=True)
        adata_Flux_labelled, _, bi_labelled, _, flux_labels, pcts_flux = fluxes
        
        # Convert flux DataFrame to PyTorch dataset for single modality training
        flux_dataset, _, _ = create_dataset.get_cls_dataset(
            data=(adata_Flux_labelled, flux_labels, bi_labelled, pcts_flux),
            batch_key=None,
            label_key=None,
            pct_key=None,
            filter_pcts=0.0,
            data_dtype=torch.float32
        )

    except FileNotFoundError as e:
        print(f"Error loading Flux data: {e}")
        raise
    # Load multimodal dataset
    try:
        with open('objects/mutlimodal_dataset.pkl', 'rb') as f:
            md = pickle.load(f)
        X, y_label, b, df_indices, pcts = md['X'], md['y_label'], md['b'], md['df_indices'], md['pcts']
        
        feature_names = list(X[0].columns) + ['batch_rna'] + list(X[1].columns) + ['batch_atac'] + list(X[2].columns) + ['batch_flux']
        y_number = torch.tensor([{'reprogramming':1, 'dead-end':0}[i] for i in list(y_label)], dtype=torch.float32)
        multimodal_dataset = create_dataset.MultiModalDataset(X, b, y_number, df_indices, pcts, y_label)
    except FileNotFoundError as e:
        print(f"Error loading multimodal data: {e}")
        raise
    
    print("Data loading completed successfully.")
    
    return {
        'rna_labelled': rna_dataset,  # Now a PyTorch dataset
        'rna_anndata': adata_RNA_labelled,  # Keep original AnnData for reference
        'atac_labelled': atac_dataset,  # Now a PyTorch dataset  
        'atac_anndata': adata_ATAC_labelled,  # Keep original AnnData for reference
        'multimodal_dataset': multimodal_dataset,
        'flux_labelled': flux_dataset,  # Now a PyTorch dataset
        'flux_dataframe': adata_Flux_labelled,  # Keep original DataFrame for shape info
        'feature_names': feature_names,
        'rna_vocab_size': rna_vocab_size,
    }

def setup_model_configs(data_dict):
    """Setup model configurations for all models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Shared config
    share_config = {
        "d_model": 128,
        "d_ff": 16,
        "n_heads": 8,
        "n_encoder_layers": 2,
        "n_batches": 3,
        "dropout_rate": 0.1,
    }
    
    # FIX: Get actual data shapes from datasets instead of non-existent 'X' key
    # Get sample data to determine shapes
    multimodal_sample = data_dict['multimodal_dataset'][0][0]  # (rna, atac, flux)
    rna_sample, atac_sample, flux_sample = multimodal_sample
    
    # Multimodal config
    model_config_rna = {
        "vocab_size": data_dict['rna_vocab_size'],
        "seq_len": rna_sample.shape[0],
    }
    model_config_atac = {
        "vocab_size": 1,
        "seq_len": atac_sample.shape[0],
    }
    model_config_flux = {
        "vocab_size": 1,
        "seq_len": flux_sample.shape[0],
    }
    model_config_multi = {
        "d_model": 128,
        "n_heads_cls": 8,
        "d_ff_cls": 16,
    }
    
    multimodal_config = {
        "Share": share_config, 
        "RNA": model_config_rna, 
        "ATAC": model_config_atac, 
        "Flux": model_config_flux, 
        "Multi": model_config_multi
    }
    
    # Single modality configs - use actual dataset shapes
    # For single modality models, get shapes from PyTorch datasets
    rna_sample_single = data_dict['rna_labelled'][0][0]  # Get first sample from RNA PyTorch dataset
    atac_sample_single = data_dict['atac_labelled'][0][0]  # Get first sample from ATAC PyTorch dataset
    flux_sample_single = data_dict['flux_labelled'][0][0]  # Get first sample from Flux PyTorch dataset
    # Also get number of features from original DataFrame for reference
    flux_n_features = data_dict['flux_dataframe'].shape[1]
    print('seq length for multi, rna, atac and flux: ', rna_sample.shape[0], atac_sample.shape[0], flux_sample.shape[0], 
          'single modality shapes:', rna_sample_single.shape[0], atac_sample_single.shape[0], flux_sample_single.shape[0], f'(flux_df_cols: {flux_n_features})')
    
    RNA_config = {
        "vocab_size": data_dict['rna_vocab_size'],
        "seq_len": rna_sample_single.shape[0],
        "d_model": 128,
        "d_ff": 16,
        "n_heads": 8,
        "n_encoder_layers": 2,
        "n_batches": 3,
        "dropout_rate": 0.2
    }
    
    ATAC_config = {
        "vocab_size": 1,
        "seq_len": atac_sample_single.shape[0],
        "d_model": 128,
        "d_ff": 16,
        "n_heads": 8,
        "n_encoder_layers": 2,
        "n_batches": 3,
        "dropout_rate": 0.2
    }
    
    Flux_config = {
        "vocab_size": 1,
        "seq_len": flux_sample_single.shape[0],
        "d_model": 128,
        "d_ff": 16,
        "n_heads": 8,
        "n_encoder_layers": 2,
        "n_batches": 3,
        "dropout_rate": 0.2
    }
    
    return {
        'Multi': multimodal_config,
        'RNA': RNA_config,
        'ATAC': ATAC_config,
        'Flux': Flux_config,
        'device': device
    }

def identify_common_samples(data_dict):
    """
    Identify samples that have both RNA and ATAC modalities available.
    These are the 163 common samples across all models.
    Creates binary lists indicating which samples are common in each modality.
    Returns modified data_dict with common sample indicators.
    """
    multimodal_dataset = data_dict['multimodal_dataset']
    rna_dataset = data_dict['rna_labelled']
    atac_dataset = data_dict['atac_labelled']

    df_indices = multimodal_dataset.get_df_indices()
    
    # Find samples where both RNA and ATAC are not None
    valid_samples = (df_indices.RNA.notna() & df_indices.ATAC.notna())
    common_indices_multimodal = df_indices[valid_samples].index.tolist()
    common_indices_rna = df_indices[valid_samples].RNA.tolist()
    common_indices_atac = df_indices[valid_samples].ATAC.tolist()

    # Get total number of samples in each modality
    total_multimodal = len(multimodal_dataset)
    total_rna = len(rna_dataset)
    total_atac = len(atac_dataset)
    
    # Create binary lists for common samples
    # For multimodal dataset
    mm_common_indics = [1 if i in common_indices_multimodal else 0 for i in range(total_multimodal)]
    
    # For RNA dataset - find which RNA samples are in the common set
    # RNA dataset is now a PyTorch dataset, need to map back to original AnnData indices
    rna_anndata = data_dict['rna_anndata']
    rna_common_indics = [1 if rna_anndata.obs.index[i] in common_indices_rna else 0 for i in range(total_rna)]
    
    # For ATAC dataset - find which ATAC samples are in the common set  
    # ATAC dataset is now a PyTorch dataset, need to map back to original AnnData indices
    atac_anndata = data_dict['atac_anndata']
    atac_common_indics = [1 if atac_anndata.obs.index[i] in common_indices_atac else 0 for i in range(total_atac)]
    
    # Get labels for common samples
    labels = [int(multimodal_dataset[i][2].item()) for i in common_indices_multimodal]
    
    # Add the common sample indicators to data_dict
    data_dict['mm_common_indics'] = mm_common_indics
    data_dict['rna_common_indics'] = rna_common_indics
    data_dict['atac_common_indics'] = atac_common_indics
    
    # Print and assert results
    print(f"Found {len(common_indices_multimodal)} common samples with both RNA and ATAC modalities")
    print(f"Common samples label distribution: {np.bincount(labels)}")
    print(f"Multimodal dataset: {total_multimodal} total samples, {sum(mm_common_indics)} common")
    print(f"RNA dataset: {total_rna} total samples, {sum(rna_common_indics)} common")
    print(f"ATAC dataset: {total_atac} total samples, {sum(atac_common_indics)} common")
    
    # Assertions
    assert len(common_indices_multimodal) == 163, f"Expected 163 common samples, got {len(common_indices_multimodal)}"
    assert sum(mm_common_indics) == 163, f"Expected 163 common samples in multimodal, got {sum(mm_common_indics)}"
    assert sum(rna_common_indics) == 163, f"Expected 163 common samples in RNA, got {sum(rna_common_indics)}"
    assert sum(atac_common_indics) == 163, f"Expected 163 common samples in ATAC, got {sum(atac_common_indics)}"
    assert len(common_indices_multimodal) == len(labels)
    
    return data_dict

def extract_metrics_from_fold_results(fold_results, model_name, seed):
    """Extract metrics from fold results and format for CSV."""
    final_records = []
    epoch_records = []
    
    for fold_result in fold_results:
        fold_num = fold_result['fold']
        
        # Extract epoch-by-epoch metrics for comprehensive CSV
        if 'epoch_metrics' in fold_result:
            for epoch, metrics in enumerate(fold_result['epoch_metrics']):
                # All samples combination
                cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
                epoch_record = {
                    'model': model_name,
                    'seed': seed,
                    'fold': fold_num,
                    'epoch': epoch + 1,
                    'combination': 'all_samples',
                    'train_loss': metrics.get('train_loss', 0),
                    'val_loss': metrics.get('val_loss', 0),
                    'train_auc': metrics.get('train_auc', 0),
                    'val_auc': metrics.get('val_auc', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'specificity': metrics.get('specificity', 0),
                    'n_samples': metrics.get('n_samples', 0),
                    'tn': cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0,
                    'fp': cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0,
                    'fn': cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0,
                    'tp': cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0,
                    'selection_criteria': fold_result.get('selection_criteria', '')
                }
                epoch_records.append(epoch_record)
                
                # Common samples combination
                if 'common_metrics' in metrics:
                    common_epoch_record = epoch_record.copy()
                    common_epoch_record['combination'] = 'common_samples'
                    common_metrics = metrics['common_metrics']
                    common_cm = common_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
                    common_epoch_record.update({
                        'val_auc': common_metrics.get('auc', 0),
                        'precision': common_metrics.get('precision', 0),
                        'recall': common_metrics.get('recall', 0),
                        'f1': common_metrics.get('f1', 0),
                        'accuracy': common_metrics.get('accuracy', 0),
                        'specificity': common_metrics.get('specificity', 0),
                        'n_samples': common_metrics.get('n_samples', 0),
                        'tn': common_cm[0][0] if len(common_cm) > 0 and len(common_cm[0]) > 0 else 0,
                        'fp': common_cm[0][1] if len(common_cm) > 0 and len(common_cm[0]) > 1 else 0,
                        'fn': common_cm[1][0] if len(common_cm) > 1 and len(common_cm[1]) > 0 else 0,
                        'tp': common_cm[1][1] if len(common_cm) > 1 and len(common_cm[1]) > 1 else 0,
                        'selection_criteria': fold_result.get('selection_criteria', '')
                    })
                    epoch_records.append(common_epoch_record)
        
        # Extract final best metrics for summary CSV
        # All samples combination
        best_cm = fold_result.get('best_confusion_matrix', [[0, 0], [0, 0]])
        final_record = {
            'model': model_name,
            'seed': seed,
            'fold': fold_num,
            'combination': 'all_samples',
            'train_auc': fold_result.get('train_auc', 0),
            'val_auc': fold_result.get('best_val_auc', 0),
            'precision': fold_result.get('best_precision', 0),
            'recall': fold_result.get('best_recall', 0),
            'f1': fold_result.get('best_f1', 0),
            'accuracy': fold_result.get('best_accuracy', 0),
            'specificity': fold_result.get('best_specificity', 0),
            'n_samples': fold_result.get('n_samples', 0),
            'tn': best_cm[0][0] if len(best_cm) > 0 and len(best_cm[0]) > 0 else 0,
            'fp': best_cm[0][1] if len(best_cm) > 0 and len(best_cm[0]) > 1 else 0,
            'fn': best_cm[1][0] if len(best_cm) > 1 and len(best_cm[1]) > 0 else 0,
            'tp': best_cm[1][1] if len(best_cm) > 1 and len(best_cm[1]) > 1 else 0,
            'model_path': fold_result.get('best_model_path', ''),
            'selection_criteria': fold_result.get('selection_criteria', '')
        }
        final_records.append(final_record)
        
        # Common samples combination
        if 'best_common_metrics' in fold_result:
            common_metrics = fold_result['best_common_metrics']
            common_best_cm = common_metrics.get('confusion_matrix', [[0, 0], [0, 0]]) if common_metrics else [[0, 0], [0, 0]]
            common_record = {
                'model': model_name,
                'seed': seed,
                'fold': fold_num,
                'combination': 'common_samples',
                'train_auc': fold_result.get('train_auc', 0),
                'val_auc': common_metrics.get('auc', 0),
                'precision': common_metrics.get('precision', 0),
                'recall': common_metrics.get('recall', 0),
                'f1': common_metrics.get('f1', 0),
                'accuracy': common_metrics.get('accuracy', 0),
                'specificity': common_metrics.get('specificity', 0),
                'n_samples': common_metrics.get('n_samples', 0),
                'tn': common_best_cm[0][0] if len(common_best_cm) > 0 and len(common_best_cm[0]) > 0 else 0,
                'fp': common_best_cm[0][1] if len(common_best_cm) > 0 and len(common_best_cm[0]) > 1 else 0,
                'fn': common_best_cm[1][0] if len(common_best_cm) > 1 and len(common_best_cm[1]) > 0 else 0,
                'tp': common_best_cm[1][1] if len(common_best_cm) > 1 and len(common_best_cm[1]) > 1 else 0,
                'model_path': fold_result.get('best_model_path', ''),
                'selection_criteria': fold_result.get('selection_criteria', '')
            }
            final_records.append(common_record)
    
    return final_records, epoch_records

def identify_multimodal_sample_types(data_dict):
    """
    Identify the three types of multimodal samples based on available modalities.
    
    Returns:
        sample_types: List of sample type indicators for each sample
            - 0: ATAC-only samples (RNA=null, ATAC=not null)
            - 1: RNA+Flux samples (RNA=not null, ATAC=null) 
            - 2: Common samples (RNA=not null, ATAC=not null)
    """
    multimodal_dataset = data_dict['multimodal_dataset']
    df_indices = multimodal_dataset.get_df_indices()
    
    sample_types = []
    for i in range(len(multimodal_dataset)):
        rna_available = df_indices.iloc[i]['RNA'] is not None and not pd.isna(df_indices.iloc[i]['RNA'])
        atac_available = df_indices.iloc[i]['ATAC'] is not None and not pd.isna(df_indices.iloc[i]['ATAC'])
        
        if not rna_available and atac_available:
            # ATAC-only samples
            sample_types.append(0)
        elif rna_available and not atac_available:
            # RNA+Flux samples
            sample_types.append(1)
        elif rna_available and atac_available:
            # Common samples (all three modalities)
            sample_types.append(2)
        else:
            # This shouldn't happen in your dataset, but handle it
            raise ValueError(f"Sample {i} has no available modalities")
    
    return sample_types

def create_compound_stratification_labels(model_name, data_dict):
    """
    Create compound stratification labels that combine sample type/common status and actual labels.
    
    For multimodal model (6 classes):
    - 0: ATAC-only samples with label 0 (dead-end)
    - 1: ATAC-only samples with label 1 (reprogramming)
    - 2: RNA+Flux samples with label 0 (dead-end)
    - 3: RNA+Flux samples with label 1 (reprogramming)
    - 4: Common samples with label 0 (dead-end)
    - 5: Common samples with label 1 (reprogramming)
    
    For other models (4 classes):
    - 0: Non-common samples with label 0 (dead-end)
    - 1: Non-common samples with label 1 (reprogramming)
    - 2: Common samples with label 0 (dead-end)
    - 3: Common samples with label 1 (reprogramming)
    
    Returns:
        compound_labels: List of compound class labels for stratification
        actual_labels: List of actual binary labels (0/1)
        sample_type_indices: List of sample type indicators
    """
    # Get dataset and extract actual labels
    if model_name == 'Multi':
        dataset = data_dict['multimodal_dataset']
        actual_labels = [int(dataset[i][2].item()) for i in range(len(dataset))]
        
        # Get multimodal sample types (0=ATAC-only, 1=RNA+Flux, 2=Common)
        sample_type_indices = identify_multimodal_sample_types(data_dict)
        
        # Create compound labels: sample_type * 2 + actual_label
        # This creates 6 classes: 0, 1, 2, 3, 4, 5
        compound_labels = []
        for i in range(len(actual_labels)):
            compound_class = sample_type_indices[i] * 2 + actual_labels[i]
            compound_labels.append(compound_class)
            
    elif model_name == 'RNA':
        common_indices = data_dict['rna_common_indics']
        dataset = data_dict['rna_labelled']
        actual_labels = [int(dataset[i][2].item()) for i in range(len(dataset))]
        sample_type_indices = common_indices
        
        # Create compound labels: common_status * 2 + actual_label (4 classes)
        compound_labels = []
        for i in range(len(actual_labels)):
            compound_class = common_indices[i] * 2 + actual_labels[i]
            compound_labels.append(compound_class)
            
    elif model_name == 'ATAC':
        common_indices = data_dict['atac_common_indics']
        dataset = data_dict['atac_labelled']
        actual_labels = [int(dataset[i][2].item()) for i in range(len(dataset))]
        sample_type_indices = common_indices
        
        # Create compound labels: common_status * 2 + actual_label (4 classes)
        compound_labels = []
        for i in range(len(actual_labels)):
            compound_class = common_indices[i] * 2 + actual_labels[i]
            compound_labels.append(compound_class)
            
    elif model_name == 'Flux':
        # Flux dataset uses same sample indices as RNA
        common_indices = data_dict['rna_common_indics']
        dataset = data_dict['flux_labelled']
        actual_labels = [int(dataset[i][2].item()) for i in range(len(dataset))]
        sample_type_indices = common_indices
        
        # Create compound labels: common_status * 2 + actual_label (4 classes)
        compound_labels = []
        for i in range(len(actual_labels)):
            compound_class = common_indices[i] * 2 + actual_labels[i]
            compound_labels.append(compound_class)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Verify lengths match
    if len(sample_type_indices) != len(actual_labels):
        raise ValueError(f"Length mismatch for {model_name}: sample_type_indices={len(sample_type_indices)}, actual_labels={len(actual_labels)}")
    
    return compound_labels, actual_labels, sample_type_indices

def get_valid_indics(model_name, data_dict, seed=42):
    """
    Create stratified cross-validation folds using compound stratification that considers
    both sample type/common status and actual labels for better balance.
    
    For multimodal: 6 classes (3 sample types × 2 labels)
    For other models: 4 classes (2 sample types × 2 labels)
    """
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    
    # Create compound stratification labels
    compound_labels, actual_labels, sample_type_indices = create_compound_stratification_labels(model_name, data_dict)
    all_indices = list(range(len(compound_labels)))
    
    # Print compound class distribution
    compound_counts = np.bincount(compound_labels)
    print(f"\nCompound class distribution for {model_name}:")
    
    if model_name == 'Multi':
        # 6 classes for multimodal
        class_names = [
            "ATAC-only + dead-end", "ATAC-only + reprogramming",
            "RNA+Flux + dead-end", "RNA+Flux + reprogramming", 
            "Common + dead-end", "Common + reprogramming"
        ]
        for i, name in enumerate(class_names):
            count = compound_counts[i] if len(compound_counts) > i else 0
            print(f"  Class {i} ({name}): {count}")
    else:
        # 4 classes for other models
        class_names = [
            "Non-common + dead-end", "Non-common + reprogramming",
            "Common + dead-end", "Common + reprogramming"
        ]
        for i, name in enumerate(class_names):
            count = compound_counts[i] if len(compound_counts) > i else 0
            print(f"  Class {i} ({name}): {count}")
    
    # Create stratified folds using compound labels
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    loop = skf.split(all_indices, compound_labels)
    
    # Analyze fold distribution and collect common indices per fold
    print(f"\nFold distribution analysis for {model_name}:")
    fold_stats = []
    common_indices_per_fold = []
    
    max_classes = 6 if model_name == 'Multi' else 4
    
    for i, (_, val_idx) in enumerate(loop):
        # Get validation samples for this fold
        val_compound_labels = [compound_labels[idx] for idx in val_idx]
        val_actual_labels = [actual_labels[idx] for idx in val_idx]
        val_sample_types = [sample_type_indices[idx] for idx in val_idx]
        
        # Count distributions
        val_compound_counts = np.bincount(val_compound_labels, minlength=max_classes)
        
        if model_name == 'Multi':
            # For multimodal: identify common samples (sample_type == 2)
            common_in_fold = [idx for idx in val_idx if sample_type_indices[idx] == 2]
            atac_only_in_fold = [idx for idx in val_idx if sample_type_indices[idx] == 0]
            rna_flux_in_fold = [idx for idx in val_idx if sample_type_indices[idx] == 1]
            
            fold_stat = {
                'fold': i + 1,
                'total_samples': len(val_idx),
                'atac_only_samples': len(atac_only_in_fold),
                'rna_flux_samples': len(rna_flux_in_fold),
                'common_samples': len(common_in_fold),
                'dead_end_samples': sum(1 for label in val_actual_labels if label == 0),
                'reprogramming_samples': sum(1 for label in val_actual_labels if label == 1),
            }
            # Add class counts
            for j in range(6):
                fold_stat[f'class_{j}'] = val_compound_counts[j]
                
            fold_stats.append(fold_stat)
            common_indices_per_fold.append(common_in_fold)
            
            print(f"  Fold {i+1}: {len(val_idx)} total | ATAC-only:{len(atac_only_in_fold)}, RNA+Flux:{len(rna_flux_in_fold)}, Common:{len(common_in_fold)}")
            print(f"    Classes [0:{val_compound_counts[0]}, 1:{val_compound_counts[1]}, 2:{val_compound_counts[2]}, 3:{val_compound_counts[3]}, 4:{val_compound_counts[4]}, 5:{val_compound_counts[5]}]")
        else:
            # For other models: identify common samples (sample_type == 1)
            common_in_fold = [idx for idx in val_idx if sample_type_indices[idx] == 1]
            
            fold_stat = {
                'fold': i + 1,
                'total_samples': len(val_idx),
                'common_samples': len(common_in_fold),
                'non_common_samples': len(val_idx) - len(common_in_fold),
                'dead_end_samples': sum(1 for label in val_actual_labels if label == 0),
                'reprogramming_samples': sum(1 for label in val_actual_labels if label == 1),
                'class_0': val_compound_counts[0],  # Non-common + dead-end
                'class_1': val_compound_counts[1],  # Non-common + reprogramming
                'class_2': val_compound_counts[2],  # Common + dead-end
                'class_3': val_compound_counts[3],  # Common + reprogramming
            }
            fold_stats.append(fold_stat)
            common_indices_per_fold.append(common_in_fold)
            
            print(f"  Fold {i+1}: {len(val_idx)} total, {len(common_in_fold)} common | "
                  f"Classes [0:{val_compound_counts[0]}, 1:{val_compound_counts[1]}, "
                  f"2:{val_compound_counts[2]}, 3:{val_compound_counts[3]}]")
    
    # Print summary statistics
    if model_name == 'Multi':
        avg_atac_only = np.mean([stat['atac_only_samples'] for stat in fold_stats])
        avg_rna_flux = np.mean([stat['rna_flux_samples'] for stat in fold_stats])
        avg_common = np.mean([stat['common_samples'] for stat in fold_stats])
        avg_dead_end = np.mean([stat['dead_end_samples'] for stat in fold_stats])
        avg_reprogramming = np.mean([stat['reprogramming_samples'] for stat in fold_stats])
        
        print(f"Average per fold: {avg_atac_only:.1f} ATAC-only, {avg_rna_flux:.1f} RNA+Flux, {avg_common:.1f} common")
        print(f"                  {avg_dead_end:.1f} dead-end, {avg_reprogramming:.1f} reprogramming")
        
        # Check balance quality
        atac_std = np.std([stat['atac_only_samples'] for stat in fold_stats])
        rna_flux_std = np.std([stat['rna_flux_samples'] for stat in fold_stats])
        common_std = np.std([stat['common_samples'] for stat in fold_stats])
        label_std = np.std([stat['dead_end_samples'] for stat in fold_stats])
        
        print(f"Balance quality (std dev): ATAC-only={atac_std:.2f}, RNA+Flux={rna_flux_std:.2f}, common={common_std:.2f}, labels={label_std:.2f}")
    else:
        avg_common = np.mean([stat['common_samples'] for stat in fold_stats])
        avg_dead_end = np.mean([stat['dead_end_samples'] for stat in fold_stats])
        avg_reprogramming = np.mean([stat['reprogramming_samples'] for stat in fold_stats])
        
        print(f"Average per fold: {avg_common:.1f} common, {avg_dead_end:.1f} dead-end, {avg_reprogramming:.1f} reprogramming")
        
        # Check balance quality
        common_std = np.std([stat['common_samples'] for stat in fold_stats])
        label_std = np.std([stat['dead_end_samples'] for stat in fold_stats])
        
        print(f"Balance quality (std dev): common={common_std:.2f}, labels={label_std:.2f}")
    
    # Reset the generator since we consumed it for analysis
    loop = skf.split(all_indices, compound_labels)
    return loop, common_indices_per_fold

def train_model_seed_fold(model_name, model_info, data_dict, 
                              epochs=10, seed=42, results_dir="analysis_results", 
                              selection_criteria='common_samples',
                              batch_size=32):
    """
    Train a model using fair validation folds (same validation samples across all models).
    """
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    
    # Create model-specific save path
    model_save_path = os.path.join(results_dir, "models", f"{model_name}_seed{seed}")
    os.makedirs(model_save_path, exist_ok=True)
    
    fold_results = []

    val_indices_loop, common_indices_per_fold = get_valid_indics(model_name, data_dict, seed=seed)
    for fold, (train_indices, val_indices) in enumerate(val_indices_loop):
        print(f'Model: {model_name}, Seed: {seed}, Fold {fold+1}/{len(common_indices_per_fold)} (val samples: {len(val_indices)}, common samples: {len(common_indices_per_fold[fold])})')

        dataset = model_info['dataset']

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        print(f"Using {len(train_indices)} training samples and {len(val_indices)} validation samples")
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Train the model for this fold
        fold_result = train_single_fold(
            model_name=model_name,
            model_config=model_info['config'],
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            val_indices=val_indices, 
            common_val_indices=common_indices_per_fold[fold],
            epochs=epochs,
            fold=fold+1,
            save_path=model_save_path,
            use_mlm=model_info['use_mlm'],
            mlm_path=model_info['mlm_path'],
            selection_criteria=selection_criteria
        )
        
        fold_results.append(fold_result)
    
    return fold_results

def train_single_fold(model_name, model_config, seed, train_loader, val_loader, 
                      val_indices, common_val_indices, epochs, fold, 
                      save_path, use_mlm=True, mlm_path=None,
                      selection_criteria='common_samples',
                      batch_size=32):
    """
    Train a single fold of a model.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
    from models.transformers import SingleTransformer
    from utils.helpers import create_multimodal_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    if model_name == 'Multi':
        model = create_multimodal_model(model_config, device, use_mlm=use_mlm)
    else:
        model = SingleTransformer(model_name, **model_config).to(device)
        if use_mlm and mlm_path:
            model.load_state_dict(torch.load(mlm_path), strict=False)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False)
    
    # Calculate pos_weight for weighted BCE loss
    train_labels = []
    for batch in train_loader:
        _, _, labels = batch
        train_labels.extend(labels.tolist())
    train_labels = torch.tensor(train_labels, dtype=torch.float)
    pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()
    pos_weight = pos_weight.to(device)
    
    # Training loop
    best_val_auc = 0.0
    best_metrics = {}
    best_common_metrics = {}
    best_model_path = None
    epoch_metrics = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for inputs, bi, y in train_loader:
            if isinstance(inputs, list):
                rna, atac, flux = inputs
                rna, atac, flux = rna.to(device), atac.to(device), flux.to(device)
                inputs = (rna, atac, flux)
            else:
                inputs = inputs.to(device)
            bi, y = bi.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds, _ = model(inputs, bi)
            preds = preds.squeeze()
            
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if y.dim() == 0:
                y = y.unsqueeze(0)
            
            loss = weighted_bce_loss(preds, y, pos_weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Store training predictions for AUC calculation
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(y.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for inputs, bi, y in val_loader:
                if isinstance(inputs, list):
                    rna, atac, flux = inputs
                    rna, atac, flux = rna.to(device), atac.to(device), flux.to(device)
                    inputs = (rna, atac, flux)
                else:
                    inputs = inputs.to(device)
                bi, y = bi.to(device), y.to(device)
                
                preds, _ = model(inputs, bi)
                preds = preds.squeeze()
                
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                
                loss = weighted_bce_loss(preds, y, pos_weight)
                val_loss += loss.item()
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        # Convert to numpy arrays
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        val_preds_binary = (val_preds > 0.5).astype(int)
        
        # Calculate training and validation metrics
        train_auc = roc_auc_score(train_labels, train_preds) if len(np.unique(train_labels)) > 1 else 0.0
        val_auc = roc_auc_score(val_labels, val_preds) if len(np.unique(val_labels)) > 1 else 0.0
        val_precision = precision_score(val_labels, val_preds_binary, zero_division=0)
        val_recall = recall_score(val_labels, val_preds_binary, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds_binary, zero_division=0)
        val_accuracy = accuracy_score(val_labels, val_preds_binary)
        
        # Calculate confusion matrix and specificity
        val_confusion_matrix = confusion_matrix(val_labels, val_preds_binary)
        tn, fp, fn, tp = val_confusion_matrix.ravel()
        val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate metrics for common samples if common_val_indices is provided
        common_metrics = None
        if common_val_indices is not None and len(common_val_indices) > 0:
            # Find common samples in validation set
            # common_val_indices should be the actual indices that are common samples in this validation fold
            # val_indices are the validation fold indices, common_val_indices should be a subset of val_indices
            
            # Convert to numpy arrays and ensure same data type
            val_indices_array = np.array(val_indices)
            common_val_indices_array = np.array(common_val_indices)
            
            val_common_mask = np.isin(val_indices_array, common_val_indices_array)
            if np.sum(val_common_mask) > 0:
                common_val_preds = val_preds[val_common_mask]
                common_val_labels = val_labels[val_common_mask]
                common_val_preds_binary = (common_val_preds > 0.5).astype(int)
                
                # Calculate confusion matrix for common samples
                common_confusion_matrix = confusion_matrix(common_val_labels, common_val_preds_binary)
                
                # Initialize common metrics
                common_metrics = {
                    'n_samples': np.sum(val_common_mask),
                    'auc': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'accuracy': 0.0,
                    'specificity': 0.0,
                    'confusion_matrix': common_confusion_matrix.tolist()  # Store as list for JSON serialization
                }
                
                # Calculate metrics only if both classes are present
                if len(np.unique(common_val_labels)) > 1:
                    common_metrics['auc'] = roc_auc_score(common_val_labels, common_val_preds)
                    common_metrics['precision'] = precision_score(common_val_labels, common_val_preds_binary, zero_division=0)
                    common_metrics['recall'] = recall_score(common_val_labels, common_val_preds_binary, zero_division=0)
                    common_metrics['f1'] = f1_score(common_val_labels, common_val_preds_binary, zero_division=0)
                    common_metrics['accuracy'] = accuracy_score(common_val_labels, common_val_preds_binary)
                    
                    # Calculate specificity for common samples
                    if common_confusion_matrix.shape == (2, 2):
                        common_tn, common_fp, common_fn, common_tp = common_confusion_matrix.ravel()
                        common_metrics['specificity'] = common_tn / (common_tn + common_fp) if (common_tn + common_fp) > 0 else 0
                else:
                    # If only one class, accuracy is still meaningful
                    common_metrics['accuracy'] = accuracy_score(common_val_labels, common_val_preds_binary)

        # Store epoch metrics
        epoch_metric = {
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'train_auc': train_auc,
            'val_auc': val_auc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'accuracy': val_accuracy,
            'specificity': val_specificity,
            'n_samples': len(val_labels),
            'confusion_matrix': val_confusion_matrix.tolist(),  # Store as list for JSON serialization
            'common_metrics': common_metrics
        }
        epoch_metrics.append(epoch_metric)
        
        # Update best model based on selection criteria
        if selection_criteria == 'common_samples' and common_metrics is not None:
            current_auc = common_metrics['auc']
        else:
            current_auc = val_auc
            
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            best_metrics = {
                'auc': val_auc,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1,
                'accuracy': val_accuracy,
                'specificity': val_specificity,
                'n_samples': len(val_labels),
                'confusion_matrix': val_confusion_matrix.tolist(),  # Store as list for JSON serialization
                'common_metrics': common_metrics
            }
            
            # Store best common metrics separately
            best_common_metrics = common_metrics if common_metrics is not None else best_metrics.copy()
            
            # Save best model
            best_model_path = os.path.join(save_path, f"best_{model_name}_seed{seed}_fold{fold}_AUC_{current_auc:.3f}.pth")
            torch.save(model.state_dict(), best_model_path)
        
        scheduler.step(current_auc)
    
    # Calculate final train AUC on best model
    model.eval()
    final_train_preds = []
    final_train_labels = []
    
    with torch.no_grad():
        for inputs, bi, y in train_loader:
            if isinstance(inputs, list):
                rna, atac, flux = inputs
                rna, atac, flux = rna.to(device), atac.to(device), flux.to(device)
                inputs = (rna, atac, flux)
            else:
                inputs = inputs.to(device)
            bi, y = bi.to(device), y.to(device)
            
            preds, _ = model(inputs, bi)
            preds = preds.squeeze()
            
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if y.dim() == 0:
                y = y.unsqueeze(0)
            
            final_train_preds.extend(preds.cpu().numpy())
            final_train_labels.extend(y.cpu().numpy())
    
    final_train_preds = np.array(final_train_preds)
    final_train_labels = np.array(final_train_labels)
    final_train_auc = roc_auc_score(final_train_labels, final_train_preds) if len(np.unique(final_train_labels)) > 1 else 0.0
    
    # Return fold results
    fold_result = {
        'model_type': model_name,
        'fold': fold,
        'val_idx': val_indices,
        'common_val_indices': common_val_indices,
        'epoch_metrics': epoch_metrics,
        'train_auc': final_train_auc,
        'selection_criteria': selection_criteria,
        'best_val_auc': best_metrics.get('auc', 0),
        'best_precision': best_metrics.get('precision', 0),
        'best_recall': best_metrics.get('recall', 0),
        'best_f1': best_metrics.get('f1', 0),
        'best_accuracy': best_metrics.get('accuracy', 0),
        'best_specificity': best_metrics.get('specificity', 0),
        'n_samples': best_metrics.get('n_samples', 0),
        'best_confusion_matrix': best_metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
        'best_model_path': best_model_path,
        'best_common_metrics': best_common_metrics
    }
    
    return fold_result

def run_comprehensive_analysis(epochs=10,
                                selection_criteria='common_samples', 
                                output_folder="analysis_results",
                                seeds=None,
                                batch_size=32,
                                ):

    if seeds is None:
        seeds = [0, 6, 42, 123, 1000]
    
    print("=" * 80)
    print("COMPREHENSIVE CELL FATE PREDICTION ANALYSIS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Selection criteria: {selection_criteria}")
    print(f"  - Output folder: {output_folder}")
    print(f"  - Seeds: {seeds}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    results_dir = setup_analysis_folder(output_folder)
    print(f"Results will be saved to: {results_dir}")
    
    # Load data
    data_dict = load_all_data()
    
    # Identify common samples (163 samples with all modalities)
    data_dict = identify_common_samples(data_dict)
    
    # Setup model configs
    configs = setup_model_configs(data_dict)
    device = configs['device']
    print(f"Using device: {device}")
    
    
    # Define models to run
    models_info = {
        'ATAC': {
            'config': configs['ATAC'],
            'dataset': data_dict['atac_labelled'],
            'use_mlm': True,
            'mlm_path': "ckp/MLM/MLM_ATAC_ValLoss0.0019.pth"
        },
        'Flux': {
            'config': configs['Flux'],
            'dataset': data_dict['flux_labelled'],
            'use_mlm': True,
            'mlm_path': "ckp/MLM/MLM_Flux_ValLoss0.1001.pth"
        },        
        'RNA': {
            'config': configs['RNA'],
            'dataset': data_dict['rna_labelled'],
            'use_mlm': True,
            'mlm_path': "ckp/MLM/MLM_RNA_ValLoss0.4277.pth"
        },
        'Multi': {
            'config': configs['Multi'],
            'dataset': data_dict['multimodal_dataset'],
            'use_mlm': True,
            'mlm_path': None
        },
    }
    
    # Storage for all results
    all_final_records = []
    all_epoch_records = []
    
    # Calculate total runs: 4 models x 5 seeds = 20 model-seed combinations
    # Each combination runs 5 folds, so 100 total folds
    # Final CSV will have 200 rows (100 folds x 2 combinations each)
    total_model_seed_combinations = len(models_info) * len(seeds)
    total_folds = total_model_seed_combinations * 5  # 5 folds per model-seed
    model_seed_count = 0
    
    # Create fair cross-validation folds using common samples
    print(f"\nRunning {total_model_seed_combinations} model-seed combinations ({total_folds} total folds)...")
    
    # Progress bar for all model-seed combinations
    with tqdm(total=total_model_seed_combinations, desc="Overall Progress", unit="model-seed") as pbar:
        for model_name, model_info in models_info.items():
            for _, seed in enumerate(seeds):
                model_seed_count += 1
                
                # Set random seeds for reproducibility
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                print(f"\n[{model_seed_count}/{total_model_seed_combinations}] Model: {model_name}, Seed: {seed} - Starting 5-fold CV...")
                
                # Update progress description
                pbar.set_description(f"{model_name}-seed{seed}")

                # Run training with fair cross-validation (5 folds)
                fold_results = train_model_seed_fold(
                    model_name=model_name,
                    model_info=model_info,
                    data_dict=data_dict,
                    epochs=epochs,
                    seed=seed,
                    results_dir=results_dir,
                    selection_criteria=selection_criteria,
                    batch_size=batch_size
                )
                
                # Extract metrics and add to records
                final_records, epoch_records = extract_metrics_from_fold_results(
                    fold_results, model_name, seed
                )
                all_final_records.extend(final_records)
                all_epoch_records.extend(epoch_records)
                
                # Save fold results as pickle
                fold_results_file = os.path.join(
                    results_dir, "fold_results", 
                    f"fold_results_{model_name}_seed{seed}_{selection_criteria}.pkl"
                )
                with open(fold_results_file, 'wb') as f:
                    pickle.dump(fold_results, f)
                
                # Print one line result for this run
                avg_auc_all = np.mean([fr.get('best_val_auc', 0) for fr in fold_results])
                avg_auc_common = 0
                if any('best_common_metrics' in fr for fr in fold_results):
                    common_aucs = [fr['best_common_metrics'].get('auc', 0) 
                                 for fr in fold_results if 'best_common_metrics' in fr]
                    avg_auc_common = np.mean(common_aucs) if common_aucs else 0
                
                print(f"✓ {model_name}-seed{seed}: AUC_all={avg_auc_all:.4f}, AUC_common={avg_auc_common:.4f}")
                
                # Update progress by 1 (one model-seed combination completed)
                pbar.update(1)
    
    # Save final results
    print(f"\n{'='*80}")
    print("SAVING FINAL RESULTS")
    print(f"{'='*80}")
    
    if all_final_records:
        # Save final results CSV
        df_final = pd.DataFrame(all_final_records)
        df_final['run_id'] = (df_final['model'] + '_seed' + df_final['seed'].astype(str) + 
                             '_fold' + df_final['fold'].astype(str) + '_' + df_final['combination'])
        df_final['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        final_results_file = os.path.join(results_dir, "metrics", "final_results.csv")
        df_final.to_csv(final_results_file, index=False)
        print(f"Final results saved to: {final_results_file}")
        
        # Save comprehensive epoch results CSV
        if all_epoch_records:
            df_epochs = pd.DataFrame(all_epoch_records)
            df_epochs['run_id'] = (df_epochs['model'] + '_seed' + df_epochs['seed'].astype(str) + 
                                  '_fold' + df_epochs['fold'].astype(str) + '_' + df_epochs['combination'])
            df_epochs['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            epoch_results_file = os.path.join(results_dir, "metrics", "comprehensive_epoch_results.csv")
            df_epochs.to_csv(epoch_results_file, index=False)
            print(f"Comprehensive epoch results saved to: {epoch_results_file}")
        
        # Save summary statistics
        summary_stats = df_final.groupby(['model', 'combination']).agg({
            'val_auc': ['mean', 'std', 'count'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'specificity': ['mean', 'std']
        }).round(4)
        
        summary_file = os.path.join(results_dir, "metrics", "summary_statistics.csv")
        summary_stats.to_csv(summary_file)
        print(f"Summary statistics saved to: {summary_file}")
        
        # Print final summary
        print(f"\nANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"Total final records: {len(df_final)}")
        print(f"Total epoch records: {len(all_epoch_records) if all_epoch_records else 0}")
        print(f"Models analyzed: {df_final['model'].nunique()}")
        print(f"Seeds used: {df_final['seed'].nunique()}")
        print(f"Folds per seed: {df_final['fold'].nunique()}")
        print(f"Combinations: {df_final['combination'].nunique()}")
        
        print(f"\nRecords per model-combination:")
        print(df_final.groupby(['model', 'combination']).size())
        
        # Print sample count verification
        print(f"\nSample Count Verification:")
        print("-" * 60)
        sample_verification = df_final.groupby(['model', 'combination']).agg({
            'n_samples': ['mean', 'std', 'min', 'max']
        }).round(2)
        print(sample_verification)
        
    else:
        print("No results to save - all runs failed!")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    EPOCHS = 15
    SELECTION_CRITERIA = 'all_samples'  # 'all_samples' or 'common_samples'
    OUTPUT_FOLDER = "analysis_results_all_stratify_multi"
    SEEDS = [0, 6, 42, 123, 1000]
    BATCH_SIZE = 32
    
    # Run the analysis
    run_comprehensive_analysis(
        epochs=EPOCHS,
        selection_criteria=SELECTION_CRITERIA,
        output_folder=OUTPUT_FOLDER,
        seeds=SEEDS,
        batch_size=BATCH_SIZE
    )
