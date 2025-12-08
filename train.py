
import os
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils.losses import MLMLoss
from utils.helpers import create_masked_input, create_multimodal_model
from collections import defaultdict
from models import SingleTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def weighted_bce_loss(preds, targets, pos_weight):
    weights = torch.where(targets == 1, pos_weight+0.3, 1)  # Assign pos_weight to positive examples
    loss = F.binary_cross_entropy(preds, targets, weight=weights, reduction='mean')
    return loss

def focal_loss(preds, targets, alpha=1.0, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
    p_t = torch.exp(-bce_loss)
    loss = alpha * (1 - p_t) ** gamma * bce_loss
    return loss.mean()

def train_mlm(model, mlm_train_loader, mlm_val_loader, device,
              mse_based=False, epochs=10, lr=1e-3, weight_decay=1e-5, 
              tune_lr= False, save_folder=None, model_type="NULL", use_multiple_gpu=False):
    """
    Train the model with masked language modeling.
    Args:
        model (nn.Module): Model to train.
        mlm_train_loader (DataLoader): Training data loader.
        mlm_val_loader (DataLoader): Validation data loader.
        device (str): Device to train on.
        mse_based (bool, optional): Flag indicating if MSE loss is used. Defaults to False.
        epochs (int, optional): Number of epochs. Defaults to 10.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        tune_lr (bool, optional): Flag indicating if learning rate is tuned. Defaults to False.
        save_folder (str, optional): Folder to save the model. Defaults to None.
        model_type (str, optional): Identifier for the model. Defaults to "NULL".
    Returns:
        list: Training losses.
        list: Validation losses
    """
    if use_multiple_gpu:
        model = nn.DataParallel(model)
        model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if tune_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)
    
    criterion = MLMLoss(mse_based=mse_based)
    mask_token = -1
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    i = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, b in tqdm(mlm_train_loader):
            x = x.to(device)
            b = b.to(device)
            x_masked, mask = create_masked_input(x, mask_token)
            mask = mask.to(device)
            x_masked = x_masked.to(device)
            optimizer.zero_grad()
            preds = model(x_masked, b, masked_lm=True)
            loss = criterion(preds, x, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            i += 1
            print(f"MLM Epoch {epoch+1}/{epochs}, Loss: {train_loss/i:.4f}", end="\r")

        train_loss /= len(mlm_train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, b in mlm_val_loader:
                x = x.to(device)
                b = b.to(device)
                x_masked, mask = create_masked_input(x, mask_token)
                mask = mask.to(device)
                x_masked = x_masked.to(device)
                preds = model(x_masked, b, masked_lm=True)
                loss = criterion(preds, x, mask)
                val_loss += loss.item()

        val_loss /= len(mlm_val_loader)
        
        if tune_lr:
            scheduler.step(val_loss)

        i = 0
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}, LR: {lr}")
        
        if save_folder is not None:
            prev_path = os.path.join(save_folder, f"MLM_{model_type}_Epoch{epoch}.pth")
            new_path = os.path.join(save_folder, f"MLM_{model_type}_Epoch{epoch+1}.pth")
            torch.save(model.state_dict(), new_path)
            if os.path.exists(prev_path):
                os.remove(prev_path)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()

    if save_folder is not None:
        save_path = os.path.join(save_folder, f"MLM_{model_type}_ValLoss{best_loss:.4f}.pth")
        torch.save(best_model, save_path)
        print(f"Model saved at {save_path}")
        
    print("Training completed.")

    return train_losses, val_losses


def train_cls(model_type, model_config, 
              dataset, k_folds=5, batch_size=32, 
              epochs=10, lr=1e-4, weight_decay=1e-3,
              use_mlm=True, mlm_path=None, save_path="ckp/CLS", 
              device='cpu', loss_fn = "w_bce", seed=42, verbose=True,
              common_indices=None, selection_criteria='all_samples'):
    """
    Train the model on downstream task.
    
    Args:
        model_type (str): Type of model ('RNA', 'ATAC', 'Flux', 'Multi')
        model_config (dict): Model configuration
        dataset: Dataset to train on
        k_folds (int): Number of cross-validation folds
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        lr (float): Learning rate
        weight_decay (float): Weight decay
        use_mlm (bool): Whether to use MLM pretraining
        mlm_path (str): Path to MLM checkpoint
        save_path (str): Path to save models
        device (str): Device to train on
        loss_fn (str): Loss function to use
        seed (int): Random seed
        verbose (bool): Whether to print detailed output
        common_indices (list): Indices of common samples across all modalities
        selection_criteria (str): Criteria for selecting best model ('all_samples' or 'common_samples')
    """
    if model_type not in ['RNA', 'ATAC', 'Flux', 'Multi']:
        raise ValueError("model_type must be one of 'RNA', 'ATAC', 'Flux', 'Multi'")
    
    if loss_fn not in ['bce', 'w_bce', 'focal']:
        raise ValueError("loss_fn must be one of 'bce', 'w_bce', 'focal'")
    strat_labels = []
    if model_type == 'Multi':
        
        for i in range(len(dataset)):
            (rna, atac, flux), _, _ = dataset[i]
            label = (
                int(torch.any(rna != 0)) * 4 +
                int(torch.any(atac != 0)) * 2 +
                int(torch.any(flux != 0))
            )
            strat_labels.append(label)
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        loop = skf.split(np.zeros(len(dataset)), strat_labels)
    else:
        for i in range(len(dataset)):
            _, _, label = dataset[i]
            strat_labels.append(label)
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        loop = kf.split(dataset, strat_labels)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(loop):
        if verbose:
            print(f'Fold {fold+1}/{k_folds}')

        if model_type == 'Multi':
            train_dist = defaultdict(int)
            val_dist = defaultdict(int)
            for idx in train_idx:
                train_dist[strat_labels[idx]] += 1
            for idx in val_idx:
                val_dist[strat_labels[idx]] += 1
            
            if verbose:
                print("Train distribution:", dict(train_dist))
                print("Val distribution:", dict(val_dist))

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        cls_train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        cls_valid_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        if model_type == 'Multi':
            model = create_multimodal_model(model_config, device, use_mlm=use_mlm)
        else:
            model = SingleTransformer(model_type, **model_config).to(device)
            if use_mlm:
                if mlm_path is None:
                    raise ValueError("If use_mlm=True then MLM path must be provided.")
                model.load_state_dict(torch.load(mlm_path), strict=False)
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False)
        
        train_labels = torch.tensor([label.item() for _, _, label in train_subset], dtype=torch.float)
        pos_weight = (len(train_labels) - train_labels.sum()) / train_labels.sum()  # Inverse of positive class proportion
        pos_weight = pos_weight.to(device)
        if verbose:
            print(f"Positive weight: {pos_weight.item():.3f}")

        # criterion = 
        criterion = nn.BCELoss()

        best_val_auc = 0.0  
        best_common_auc = 0.0
        best_model_path = None
        fold_metrics = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': [], 'latent_auc_train': [], 'latent_auc_test': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_accuracy': [], 'val_specificity': []}
        epoch_metrics = []  # Store metrics for each epoch   

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for inputs, bi, y in cls_train_loader:
                if isinstance(inputs, list):
                    rna, atac, flux = inputs
                    rna, atac, flux = rna.to(device), atac.to(device), flux.to(device)
                    inputs = (rna, atac, flux)
                else:
                    inputs = inputs.to(device) 
                bi, y = bi.to(device), y.to(device)
                optimizer.zero_grad()
                preds, _ = model(inputs, bi) # ignore cls_output (latent)
                preds = preds.squeeze()
                # Ensure preds and y have compatible shapes for BCE loss
                if preds.dim() == 0:  # scalar
                    preds = preds.unsqueeze(0)
                if y.dim() == 0:  # scalar
                    y = y.unsqueeze(0)
                assert preds.shape == y.shape, f"{preds.shape} != {y.shape} prediction vs ground-truth"

                if loss_fn == "bce":
                    loss = criterion(preds, y)
                elif loss_fn == "w_bce":
                    loss = weighted_bce_loss(preds, y, pos_weight)#criterion(preds.squeeze(), y)
                elif loss_fn == "focal":
                    loss = focal_loss(preds, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(cls_train_loader)

            model.eval()

            train_preds, train_labels = [], []
            with torch.no_grad():
                for inputs, bi, y in cls_train_loader:
                    if isinstance(inputs, list):
                        rna, atac, flux = inputs
                        rna, atac, flux = rna.to(device), atac.to(device), flux.to(device)
                        inputs = (rna, atac, flux)
                    else:
                        inputs = inputs.to(device)
                    bi = bi.to(device)
                    preds, _ = model(inputs, bi)
                    train_preds.append(preds.cpu().numpy())
                    train_labels.append(y.cpu().numpy())

            val_loss, val_preds, val_labels, val_strat_labels = 0, [], [], []
            with torch.no_grad():
                for inputs, bi, y in cls_valid_loader:
                    if isinstance(inputs, list):
                        rna= inputs[0].to(device)
                        atac = inputs[1].to(device)
                        flux = inputs[2].to(device)
                        inputs = (rna, atac, flux)
                    else:
                        inputs = inputs.to(device)
                    bi, y = bi.to(device), y.to(device)

                    preds, _ = model(inputs, bi)
                    preds = preds.squeeze()
                    # Ensure preds and y have compatible shapes for BCE loss
                    if preds.dim() == 0:  # scalar
                        preds = preds.unsqueeze(0)
                    if y.dim() == 0:  # scalar
                        y = y.unsqueeze(0)
                    loss = criterion(preds, y)
                    val_loss += loss.item()
                    val_preds.append(preds.cpu().numpy())
                    val_labels.append(y.cpu().numpy())
                    if model_type == 'Multi':
                        strat_label = (
                        (torch.any(rna != 0, dim=1) * 4 +
                        torch.any(atac != 0, dim=1) * 2 +
                        torch.any(flux != 0, dim=1)).cpu().numpy()
                        )
                        val_strat_labels.extend(strat_label)

            val_preds = np.concatenate(val_preds).ravel()
            val_labels = np.concatenate(val_labels).ravel()
            train_preds = np.concatenate(train_preds).ravel()
            train_labels = np.concatenate(train_labels).ravel()
            if model_type == 'Multi':
                val_strat_labels = np.array(val_strat_labels)
            
            val_auc = roc_auc_score(val_labels, val_preds)
            train_auc = roc_auc_score(train_labels, train_preds)
            
            val_loss /= len(cls_valid_loader)

            val_preds_binary = (val_preds > 0.5).astype(int)
            val_precision = precision_score(val_labels, val_preds_binary)
            val_recall = recall_score(val_labels, val_preds_binary)
            val_f1 = f1_score(val_labels, val_preds_binary)
            val_accuracy = accuracy_score(val_labels, val_preds_binary)
            
            # Calculate specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(val_labels, val_preds_binary).ravel()
            val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Calculate metrics for common samples if common_indices is provided
            common_metrics = None
            if common_indices is not None:
                # Find common samples in validation set
                val_common_mask = np.isin(val_idx, common_indices)
                if np.sum(val_common_mask) > 0:
                    common_val_preds = val_preds[val_common_mask]
                    common_val_labels = val_labels[val_common_mask]
                    common_val_preds_binary = (common_val_preds > 0.5).astype(int)
                    
                    # Initialize common metrics
                    common_metrics = {
                        'n_samples': np.sum(val_common_mask),
                        'auc': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'accuracy': 0.0,
                        'specificity': 0.0
                    }
                    
                    # Calculate metrics only if both classes are present
                    if len(np.unique(common_val_labels)) > 1:
                        common_metrics['auc'] = roc_auc_score(common_val_labels, common_val_preds)
                        common_metrics['precision'] = precision_score(common_val_labels, common_val_preds_binary, zero_division=0)
                        common_metrics['recall'] = recall_score(common_val_labels, common_val_preds_binary, zero_division=0)
                        common_metrics['f1'] = f1_score(common_val_labels, common_val_preds_binary, zero_division=0)
                        common_metrics['accuracy'] = accuracy_score(common_val_labels, common_val_preds_binary)
                        
                        # Calculate specificity for common samples
                        common_cm = confusion_matrix(common_val_labels, common_val_preds_binary)
                        if common_cm.shape == (2, 2):
                            common_tn, common_fp, common_fn, common_tp = common_cm.ravel()
                            common_metrics['specificity'] = common_tn / (common_tn + common_fp) if (common_tn + common_fp) > 0 else 0
                    else:
                        # If only one class, accuracy is still meaningful
                        common_metrics['accuracy'] = accuracy_score(common_val_labels, common_val_preds_binary)

            # Calculate metrics for different modality combinations (only for Multi)
            modality_metrics = {}
            if model_type == 'Multi':
                # Define modality combinations: RNA=4, ATAC=2, Flux=1
                modality_combinations = {
                    'all_modal': 7,      # RNA + ATAC + Flux (4+2+1)
                    'rna_flux': 5,       # RNA + Flux (4+1)
                    'atac_only': 2       # ATAC only (2)
                }
                
                for combo_name, strat_value in modality_combinations.items():
                    mask = val_strat_labels == strat_value
                    n_samples = np.sum(mask)
                    
                    if n_samples > 0:
                        combo_preds = val_preds[mask]
                        combo_labels = val_labels[mask]
                        combo_preds_binary = (combo_preds > 0.5).astype(int)
                        
                        # Initialize metrics with defaults
                        metrics = {
                            'n_samples': n_samples,
                            'auc': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'f1': 0.0,
                            'accuracy': 0.0,
                            'specificity': 0.0
                        }
                        
                        # Calculate metrics only if both classes are present
                        if len(np.unique(combo_labels)) > 1:
                            metrics['auc'] = roc_auc_score(combo_labels, combo_preds)
                            metrics['precision'] = precision_score(combo_labels, combo_preds_binary, zero_division=0)
                            metrics['recall'] = recall_score(combo_labels, combo_preds_binary, zero_division=0)
                            metrics['f1'] = f1_score(combo_labels, combo_preds_binary, zero_division=0)
                            metrics['accuracy'] = accuracy_score(combo_labels, combo_preds_binary)
                            
                            # Calculate specificity
                            combo_cm = confusion_matrix(combo_labels, combo_preds_binary)
                            if combo_cm.shape == (2, 2):
                                combo_tn, combo_fp, combo_fn, combo_tp = combo_cm.ravel()
                                metrics['specificity'] = combo_tn / (combo_tn + combo_fp) if (combo_tn + combo_fp) > 0 else 0
                        else:
                            # If only one class, accuracy is still meaningful
                            metrics['accuracy'] = accuracy_score(combo_labels, combo_preds_binary)
                        
                        modality_metrics[combo_name] = metrics

            # Store epoch metrics
            epoch_metric = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_auc': train_auc,
                'val_auc': val_auc,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1,
                'accuracy': val_accuracy,
                'specificity': val_specificity,
                'n_samples': len(val_labels)
            }
            if common_metrics is not None:
                epoch_metric['common_metrics'] = common_metrics
            epoch_metrics.append(epoch_metric)

            # Determine which AUC to use for model selection
            selection_auc = val_auc
            if selection_criteria == 'common_samples' and common_metrics is not None:
                selection_auc = common_metrics['auc']
            
            # Update best model based on selection criteria
            is_best = False
            if selection_criteria == 'all_samples':
                is_best = val_auc > best_val_auc
                if is_best:
                    best_val_auc = val_auc
            elif selection_criteria == 'common_samples' and common_metrics is not None:
                is_best = common_metrics['auc'] > best_common_auc
                if is_best:
                    best_common_auc = common_metrics['auc']
                    best_val_auc = val_auc  # Still store the all samples AUC
            
            if is_best:
                best_train_auc = train_auc
                best_precision = val_precision
                best_recall = val_recall
                best_f1 = val_f1
                best_accuracy = val_accuracy
                best_specificity = val_specificity
                best_modality_metrics = deepcopy(modality_metrics) if modality_metrics else {}
                best_common_metrics = deepcopy(common_metrics) if common_metrics else None
                pre = 'NoMLM' if not use_mlm else 'wMLM'
                best_model_path = os.path.join(save_path, f"CLS_{model_type}_{pre}_fold{fold+1}_Best{selection_criteria.title()}AUC_{selection_auc:.3f}.pth")
                best_state_dict = deepcopy(model.state_dict())
                
           
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_auc)
            
            if verbose:
                print(f"\nEpoch {epoch+1}/{epochs} (lr: {current_lr:.2e})")
                print("-" * 80)
                
                # Train metrics
                print(f"{'Train':<12} (n={len(train_labels):>4}): "
                      f"L:{train_loss:>6.4f} | AUC:{train_auc:>6.4f} | P:{0:>6.4f} | R:{0:>6.4f} | "
                      f"F1:{0:>6.4f} | Acc:{0:>6.4f} | Sp:{0:>6.4f}")
                
                # Validation all samples
                print(f"{'Val-All':<12} (n={len(val_labels):>4}): "
                      f"L:{val_loss:>6.4f} | AUC:{val_auc:>6.4f} | P:{val_precision:>6.4f} | R:{val_recall:>6.4f} | "
                      f"F1:{val_f1:>6.4f} | Acc:{val_accuracy:>6.4f} | Sp:{val_specificity:>6.4f}")
                
                # Common samples metrics (if available)
                if common_metrics is not None:
                    print(f"{'Val-Common':<12} (n={common_metrics['n_samples']:>4}): "
                          f"L:{'-':>6} | AUC:{common_metrics['auc']:>6.4f} | P:{common_metrics['precision']:>6.4f} | R:{common_metrics['recall']:>6.4f} | "
                          f"F1:{common_metrics['f1']:>6.4f} | Acc:{common_metrics['accuracy']:>6.4f} | Sp:{common_metrics['specificity']:>6.4f}")
                
                # Modality-specific metrics (only for Multi)
                if model_type == 'Multi':
                    modality_names = {
                        'all_modal': 'Val-AllMod',
                        'rna_flux': 'Val-RnaFlx',
                        'atac_only': 'Val-AtacOn'
                    }
                    
                    for combo_name, display_name in modality_names.items():
                        if combo_name in modality_metrics:
                            m = modality_metrics[combo_name]
                            print(f"{display_name:<12} (n={m['n_samples']:>4}): "
                                  f"L:{'-':>6} | AUC:{m['auc']:>6.4f} | P:{m['precision']:>6.4f} | R:{m['recall']:>6.4f} | "
                                  f"F1:{m['f1']:>6.4f} | Acc:{m['accuracy']:>6.4f} | Sp:{m['specificity']:>6.4f}")
                        else:
                            print(f"{display_name:<12} (n={0:>4}): No samples")
            

            fold_metrics['train_loss'].append(train_loss)
            fold_metrics['val_loss'].append(val_loss)
            fold_metrics['train_auc'].append(train_auc)
            fold_metrics['val_auc'].append(val_auc)
            fold_metrics['val_precision'].append(val_precision)
            fold_metrics['val_recall'].append(val_recall)
            fold_metrics['val_f1'].append(val_f1)
            fold_metrics['val_accuracy'].append(val_accuracy)
            fold_metrics['val_specificity'].append(val_specificity)


        torch.save(best_state_dict, best_model_path)
        if verbose:
            print(f"\n{'='*80}")
            print(f"FOLD {fold+1} BEST RESULTS")
            print(f"{'='*80}")
            print(f"Model saved: {best_model_path}")
            print(f"{'Val-All':<12}: AUC:{best_val_auc:.4f}, P:{best_precision:.4f}, R:{best_recall:.4f}, "
                  f"F1:{best_f1:.4f}, Acc:{best_accuracy:.4f}, Sp:{best_specificity:.4f}")
            
            # Report best common samples metrics if available
            if best_common_metrics:
                m = best_common_metrics
                print(f"{'Val-Common':<12}: AUC:{m['auc']:.4f}, P:{m['precision']:.4f}, R:{m['recall']:.4f}, "
                      f"F1:{m['f1']:.4f}, Acc:{m['accuracy']:.4f}, Sp:{m['specificity']:.4f} (n={m['n_samples']})")
            
            # Report best modality-specific metrics if available
            if best_modality_metrics:
                modality_names = {
                    'all_modal': 'Val-AllMod',
                    'rna_flux': 'Val-RnaFlx', 
                    'atac_only': 'Val-AtacOn'
                }
                
                for combo_name, display_name in modality_names.items():
                    if combo_name in best_modality_metrics:
                        m = best_modality_metrics[combo_name]
                        print(f"{display_name:<12}: AUC:{m['auc']:.4f}, P:{m['precision']:.4f}, R:{m['recall']:.4f}, "
                              f"F1:{m['f1']:.4f}, Acc:{m['accuracy']:.4f}, Sp:{m['specificity']:.4f} (n={m['n_samples']})")

        # Prepare fold results
        fold_result = {
            'model_type': model_type,
            'fold': fold+1,
            'val_idx': val_idx,
            'metrics': fold_metrics,
            'epoch_metrics': epoch_metrics,  # Add epoch-by-epoch metrics
            'train_auc': best_train_auc,
            'best_val_auc': best_val_auc,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_f1': best_f1,
            'best_accuracy': best_accuracy,
            'best_specificity': best_specificity,
            'n_samples': len(val_subset),  # Add number of validation samples
            'best_model_path': best_model_path
        }
        
        # Add modality-specific metrics if available
        if best_modality_metrics:
            fold_result['best_modality_metrics'] = best_modality_metrics
        
        # Add common samples metrics if available
        if best_common_metrics:
            fold_result['best_common_metrics'] = best_common_metrics
            
        fold_results.append(fold_result)
    
    # Print cross-fold averages
    if verbose:
        print_cross_fold_summary(fold_results, model_type)
    
    return fold_results


def print_cross_fold_summary(fold_results, model_type):
    """Print cross-fold summary for all validation subsets."""
    print(f"\n{'='*100}")
    print(f"CROSS-FOLD SUMMARY ({len(fold_results)} folds)")
    print(f"{'='*100}")
    
    # Collect metrics for all samples validation
    all_metrics = {
        'auc': [fr['best_val_auc'] for fr in fold_results],
        'precision': [fr['best_precision'] for fr in fold_results],
        'recall': [fr['best_recall'] for fr in fold_results],
        'f1': [fr['best_f1'] for fr in fold_results],
        'accuracy': [fr['best_accuracy'] for fr in fold_results],
        'specificity': [fr['best_specificity'] for fr in fold_results]
    }
    
    # Print all samples summary
    print(f"{'Val-All':<12}: ", end="")
    for metric, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper()[:3]}:{mean_val:>6.4f}±{std_val:<5.4f} ", end="")
    print()
    
    # Print common samples summary (if available)
    common_metrics_available = any('best_common_metrics' in fr for fr in fold_results)
    if common_metrics_available:
        common_metrics = {
            'auc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': [],
            'specificity': []
        }
        total_common_samples = 0
        valid_folds = 0
        
        for fr in fold_results:
            if 'best_common_metrics' in fr:
                m = fr['best_common_metrics']
                for metric in common_metrics:
                    common_metrics[metric].append(m[metric])
                total_common_samples += m['n_samples']
                valid_folds += 1
        
        if valid_folds > 0:
            print(f"{'Val-Common':<12}: ", end="")
            for metric, values in common_metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values) if len(values) > 1 else 0
                print(f"{metric.upper()[:3]}:{mean_val:>6.4f}±{std_val:<5.4f} ", end="")
            print(f"(n={total_common_samples}, {valid_folds}/{len(fold_results)} folds)")
    
    # Print modality-specific summaries (only for Multi)
    if model_type == 'Multi':
        modality_combinations = ['all_modal', 'rna_flux', 'atac_only']
        modality_names = {
            'all_modal': 'Val-AllMod',
            'rna_flux': 'Val-RnaFlx',
            'atac_only': 'Val-AtacOn'
        }
        
        for combo_name in modality_combinations:
            # Collect metrics for this modality combination
            combo_metrics = {
                'auc': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'accuracy': [],
                'specificity': []
            }
            total_samples = 0
            valid_folds = 0
            
            for fr in fold_results:
                if 'best_modality_metrics' in fr and combo_name in fr['best_modality_metrics']:
                    m = fr['best_modality_metrics'][combo_name]
                    if m['n_samples'] > 0:
                        for metric in combo_metrics:
                            combo_metrics[metric].append(m[metric])
                        total_samples += m['n_samples']
                        valid_folds += 1
            
            # Print summary for this modality combination
            display_name = modality_names[combo_name]
            if valid_folds > 0:
                print(f"{display_name:<12}: ", end="")
                for metric, values in combo_metrics.items():
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values) if len(values) > 1 else 0
                        print(f"{metric.upper()[:3]}:{mean_val:>6.4f}±{std_val:<5.4f} ", end="")
                    else:
                        print(f"{metric.upper()[:3]}:{0:>6.4f}±{0:<5.4f} ", end="")
                print(f"(n={total_samples}, {valid_folds}/{len(fold_results)} folds)")
            else:
                print(f"{display_name:<12}: No samples across folds")
    
    print(f"{'='*100}")


def summarize_complete_modality_results(fold_results):
    """
    Summarize complete modality metrics across all folds.
    Args:
        fold_results: List of fold results from train_cls
    Returns:
        dict: Summary statistics for complete modality metrics
    """
    complete_metrics = []
    total_complete_samples = 0
    
    for fold_result in fold_results:
        if 'best_complete_metrics' in fold_result:
            metrics = fold_result['best_complete_metrics']
            complete_metrics.append(metrics)
            total_complete_samples += metrics['n_samples']
    
    if not complete_metrics:
        return {"message": "No complete modality samples found across folds"}
    
    # Calculate mean and std for each metric
    metrics_summary = {}
    metric_names = ['auc', 'precision', 'recall', 'f1', 'accuracy', 'specificity']
    
    for metric in metric_names:
        values = [m[metric] for m in complete_metrics]
        metrics_summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    metrics_summary['total_complete_samples'] = total_complete_samples
    metrics_summary['n_folds_with_complete'] = len(complete_metrics)
    
    return metrics_summary


def print_complete_modality_summary(fold_results):
    """Print a formatted summary of complete modality results."""
    summary = summarize_complete_modality_results(fold_results)
    
    if "message" in summary:
        print(summary["message"])
        return
    
    print(f"\n{'='*60}")
    print("COMPLETE MODALITY SAMPLES SUMMARY")
    print(f"{'='*60}")
    print(f"Folds with complete samples: {summary['n_folds_with_complete']}/{len(fold_results)}")
    print(f"Total complete samples: {summary['total_complete_samples']}")
    print()
    
    print("Complete Modality Performance (Mean ± Std):")
    print("-" * 45)
    metric_abbrev = {'auc': 'AUC', 'precision': 'P', 'recall': 'R', 
                     'f1': 'F1', 'accuracy': 'Acc', 'specificity': 'Sp'}
    
    for metric, abbrev in metric_abbrev.items():
        mean_val = summary[metric]['mean']
        std_val = summary[metric]['std']
        print(f"{abbrev:>3}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"{'='*60}")


