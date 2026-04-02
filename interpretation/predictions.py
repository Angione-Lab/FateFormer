"""
Validation Results Analysis
This module provides functions to create comprehensive DataFrames containing
sample-level predictions, labels, and metadata from cross-validation results.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from utils.helpers import create_multimodal_model
from models import SingleTransformer

def get_sample_predictions_dataframe(
    model_type, 
    multimodal_dataset, 
    fold_results, 
    model_config, 
    device='cpu',
    batch_size=32,
    adata_rna=None,
    adata_atac=None,
    threshold=0.5
):
    """
    Creates a comprehensive DataFrame with sample-level predictions and metadata.
    
    Parameters
    ----------
    model_type : str
        Type of model: 'Multi', 'RNA', 'ATAC', or 'Flux'
    multimodal_dataset : MultiModalDataset
        The multimodal dataset containing all samples
    fold_results : list
        List of fold result dictionaries from cross-validation
    model_config : dict
        Model configuration dictionary
    device : str, optional
        Device to run predictions on ('cpu', 'cuda', 'mps')
    batch_size : int, optional
        Batch size for predictions
    adata_rna : AnnData, optional
        RNA AnnData object for additional metadata
    adata_atac : AnnData, optional
        ATAC AnnData object for additional metadata
    threshold : float, optional
        Classification threshold for binary predictions (default: 0.5)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ind: Sample index in the dataset
        - fold: Fold number
        - label_numeric: Actual label (0 or 1)
        - label: Actual label name ('dead-end' or 'reprogramming')
        - predicted_value: Predicted probability [0, 1]
        - predicted_class_numeric: Predicted class (0 or 1)
        - predicted_class: Predicted class name ('dead-end' or 'reprogramming')
        - correct: Whether prediction matches label
        - abs_error: Absolute error of prediction
        - modality: Available modalities for this sample (e.g., 'RAF', 'A', 'RF')
        - batch_no: Batch number
        - pct: Percentage metadata (if available)
        - clone_size: Clone size (if available)
        - clone_id: Clone ID (if available)
        - (additional RNA/ATAC metadata if adata objects provided)
    """
    
    # Collect all predictions across folds
    all_predictions = []
    all_labels = []
    all_indices = []
    all_folds = []
    
    print(f"Processing {len(fold_results)} folds...")
    
    for fold_idx, fold in enumerate(fold_results):
        model_path = fold['best_model_path']
        val_idx = fold['val_idx']
        
        # Create validation subset
        val_subset = Subset(multimodal_dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Load model
        if model_type == 'Multi':
            model = create_multimodal_model(model_config, device, use_mlm=False)
        else:
            model = SingleTransformer(id=model_type, **model_config).to(device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Get predictions
        fold_preds = []
        fold_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, b, y = batch
                
                if isinstance(x, list):
                    rna = x[0].to(device)
                    atac = x[1].to(device)
                    flux = x[2].to(device)
                    x = (rna, atac, flux)
                else:
                    x = x.to(device)
                
                b = b.to(device)
                
                # Get predictions
                preds, _ = model(x, b)
                preds = preds.squeeze()
                
                if preds.dim() == 0:
                    preds = preds.unsqueeze(0)
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                
                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(y.numpy())
        
        # Store results
        all_predictions.extend(fold_preds)
        all_labels.extend(fold_labels)
        all_indices.extend(val_idx)
        all_folds.extend([fold_idx + 1] * len(val_idx))
        
        print(f"  Fold {fold_idx + 1}: {len(val_idx)} samples processed")
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)
    all_folds = np.array(all_folds)
    
    # Determine modality availability for each sample
    modalities = _get_modality_info(multimodal_dataset, all_indices)
    
    # Get additional metadata
    df_indices = multimodal_dataset.df_indics if hasattr(multimodal_dataset, 'df_indics') else None
    pcts = multimodal_dataset.pcts if hasattr(multimodal_dataset, 'pcts') else None
    label_names = multimodal_dataset.label_names if hasattr(multimodal_dataset, 'label_names') else None
    
    # Build base dataframe
    samples_data = []
    
    for i, (idx, pred, label, fold) in enumerate(zip(all_indices, all_predictions, all_labels, all_folds)):
        # Compute error
        abs_error = abs(label - pred)
        
        # Determine if correct
        pred_class = int(pred >= threshold)
        is_correct = pred_class == int(label)
        
        # Get batch number
        batch_no = int(multimodal_dataset.batch_no[idx].item())
        
        # Base sample info
        sample_info = {
            'ind': idx,
            'fold': fold,
            'label_numeric': int(label),
            'label': 'reprogramming' if label == 1 else 'dead-end',
            'predicted_value': float(pred),
            'predicted_class_numeric': pred_class,
            'predicted_class': 'reprogramming' if pred_class == 1 else 'dead-end',
            'correct': int(is_correct),
            'abs_error': float(abs_error),
            'modality': modalities[i],
            'batch_no': batch_no,
        }
        
        # Add percentage if available
        if pcts is not None:
            sample_info['pct'] = float(pcts[idx])
        
        # Add additional metadata from AnnData objects if available
        if df_indices is not None and (adata_rna is not None or adata_atac is not None):
            rna_id = df_indices.iloc[idx, 0] if df_indices.shape[1] > 0 else None
            atac_id = df_indices.iloc[idx, 1] if df_indices.shape[1] > 1 else None
            
            # Try to get metadata from RNA or ATAC
            metadata_added = False
            
            if adata_rna is not None and rna_id is not None and rna_id in adata_rna.obs.index:
                obs = adata_rna.obs.loc[rna_id]
                _add_obs_metadata(sample_info, obs)
                metadata_added = True
            
            if not metadata_added and adata_atac is not None and atac_id is not None and atac_id in adata_atac.obs.index:
                obs = adata_atac.obs.loc[atac_id]
                _add_obs_metadata(sample_info, obs)
        
        samples_data.append(sample_info)
    
    # Create DataFrame
    df_samples = pd.DataFrame(samples_data)
    
    # Sort by index for easier analysis
    df_samples = df_samples.sort_values('ind').reset_index(drop=True)
    
    print(f"\nTotal samples: {len(df_samples)}")
    print(f"Correct predictions: {df_samples['correct'].sum()} ({100 * df_samples['correct'].mean():.2f}%)")
    print(f"Mean absolute error: {df_samples['abs_error'].mean():.4f}")
    
    return df_samples


def _get_modality_info(dataset, indices):
    """
    Determine which modalities are available for each sample.
    
    Returns a list of modality strings:
    - 'RAF': RNA, ATAC, Flux all available
    - 'RA': RNA and ATAC available
    - 'RF': RNA and Flux available  
    - 'AF': ATAC and Flux available
    - 'R': Only RNA available
    - 'A': Only ATAC available
    - 'F': Only Flux available
    """
    modalities = []
    
    for idx in indices:
        # Check if each modality has data
        has_rna = (dataset.rna_data[idx] != 0).any().item()
        has_atac = (dataset.atac_data[idx] != 0).any().item()
        has_flux = (dataset.flux_data[idx] != 0).any().item()
        
        # Build modality string
        modality = ''
        if has_rna:
            modality += 'R'
        if has_atac:
            modality += 'A'
        if has_flux:
            modality += 'F'
        
        modalities.append(modality if modality else 'None')
    
    return modalities

def _add_obs_metadata(sample_info, obs):
    """Add metadata from AnnData obs to sample_info dictionary."""
    metadata_fields = [
        'clone_size', 'clone_id', 'cells_RNA', 'cells_ATAC',
        'cells_RNA_D3', 'cells_ATAC_D3', 'n_genes', 'phase',
        'G2M_score', 'pct_counts_mt', 'total_counts'
    ]
    
    for field in metadata_fields:
        if field in obs:
            value = obs[field]
            # Handle different data types
            if pd.notna(value):
                if isinstance(value, (int, float, np.integer, np.floating)):
                    sample_info[field] = value
                else:
                    sample_info[field] = str(value)

def summarize_by_modality(df_samples):
    """
    Summarize prediction performance by modality.
    
    Parameters
    ----------
    df_samples : pd.DataFrame
        DataFrame from get_sample_predictions_dataframe
    
    Returns
    -------
    pd.DataFrame
        Summary statistics grouped by modality
    """
    summary = df_samples.groupby('modality').agg({
        'ind': 'count',
        'correct': 'mean',
        'abs_error': 'mean',
        'predicted_value': ['mean', 'std']
    }).round(4)
    
    summary.columns = ['n_samples', 'accuracy', 'mean_abs_error', 'mean_pred', 'std_pred']
    summary = summary.reset_index()
    summary = summary.sort_values('n_samples', ascending=False)
    
    return summary

def summarize_by_fold(df_samples):
    """
    Summarize prediction performance by fold.
    
    Parameters
    ----------
    df_samples : pd.DataFrame
        DataFrame from get_sample_predictions_dataframe
    
    Returns
    -------
    pd.DataFrame
        Summary statistics grouped by fold
    """
    summary = df_samples.groupby('fold').agg({
        'ind': 'count',
        'correct': 'mean',
        'abs_error': 'mean',
        'predicted_value': ['mean', 'std']
    }).round(4)
    
    summary.columns = ['n_samples', 'accuracy', 'mean_abs_error', 'mean_pred', 'std_pred']
    summary = summary.reset_index()
    
    return summary
def get_misclassified_samples(df_samples):
    """
    Get only misclassified samples.
    
    Parameters
    ----------
    df_samples : pd.DataFrame
        DataFrame from get_sample_predictions_dataframe
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing only misclassified samples
    """
    return df_samples[df_samples['correct'] == 0].copy()
def get_samples_by_modality(df_samples, modality):
    """
    Get samples filtered by modality.
    
    Parameters
    ----------
    df_samples : pd.DataFrame
        DataFrame from get_sample_predictions_dataframe
    modality : str
        Modality string (e.g., 'RAF', 'A', 'RF')
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    return df_samples[df_samples['modality'] == modality].copy()

