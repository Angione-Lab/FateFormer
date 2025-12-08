import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import SingleTransformer
from utils.helpers import create_multimodal_model
from data.create_dataset import MultiModalDataset
from .attentions import filter_idx

def get_latent_space(id, fold_results, labelled_dataset, 
            model_config, device, batch_size=32, common_samples=True):

    if id not in ['RNA', 'ATAC', 'Flux', 'Multi']:
        raise ValueError("id must be one of 'RNA', 'ATAC', 'Flux', 'Multi'")

    latent_space = []
    labels = []
    preds = []
    for fold in fold_results:
        model_path = fold['best_model_path']
        val_idx = fold['val_idx']
        if common_samples:
            val_idx = filter_idx(labelled_dataset, val_idx)
        val_ds = Subset(labelled_dataset, val_idx)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        if id=='Multi':
            model = create_multimodal_model(model_config, device, use_mlm=False)
        else:
            model = SingleTransformer(id=id, **model_config).to(device)

        # Load weights to CPU first, then move to target device (handles CUDA->MPS/CPU transfer)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, b, y = batch
                if isinstance(x, list):
                    rna= x[0].to(device)
                    atac = x[1].to(device)
                    flux = x[2].to(device)
                    x = (rna, atac, flux)
                else:
                    x = x.to(device)
                b = b.to(device)
                
                ls, pred = model.get_latent_space(x, b)
                latent_space.append(ls.cpu().numpy())
                labels.append(y.numpy())
                preds.append(pred.cpu().numpy())
    latent_space = np.concatenate(latent_space)
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    preds = np.round(preds)
    return latent_space, labels, preds

def get_latent_space_cached(models, fold_results, dataset, device, batch_size=64, common_samples=True):
    """
    Compute latent space using preloaded models.
    """
    latent_space = []
    labels = []
    preds = []
    for model, fold in zip(models, fold_results):
        val_idx = fold['val_idx']
        if common_samples:
            val_idx = filter_idx(dataset, val_idx)
        val_ds = Subset(dataset, val_idx)
        # Increase batch size to speed up inference
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, b, y = batch
                if isinstance(x, list):
                    # For multimodal inputs, move each modality to device
                    rna = x[0].to(device)
                    atac = x[1].to(device)
                    flux = x[2].to(device)
                    x = (rna, atac, flux)
                else:
                    x = x.to(device)
                b = b.to(device)
                ls, pred = model.get_latent_space(x, b)
                latent_space.append(ls.cpu().numpy())
                labels.append(y.numpy())
                preds.append(pred.cpu().numpy())
    latent_space = np.concatenate(latent_space)
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    preds = np.round(preds)
    return latent_space, labels, preds

def measure_shift(original_latent, perturbed_latent):
    return np.mean(np.linalg.norm(original_latent - perturbed_latent, axis=1))

def perturb_feature(data, feature_idx, perturbation_type='additive', scale=0.1, min_samples_threshold=10):
    perturbed_data = data.clone()
    non_zero_rows_mask = data[:, feature_idx] != 0
    
    # Check if feature has enough non-zero samples
    if non_zero_rows_mask.sum() < min_samples_threshold:
        return None, True  # Return None and flag indicating insufficient samples

    if perturbation_type == 'shuffle':
        # Shuffle only non-zero values (preserves sparsity pattern)
        non_zero_values = perturbed_data[non_zero_rows_mask, feature_idx].clone()
        shuffled_idx = torch.randperm(non_zero_values.size(0), device=perturbed_data.device)
        perturbed_data[non_zero_rows_mask, feature_idx] = non_zero_values[shuffled_idx]
        
    elif perturbation_type == 'shuffle_all':
        # Shuffle all values (including zeros)
        shuffled_idx = torch.randperm(perturbed_data.size(0), device=perturbed_data.device)
        perturbed_data[:, feature_idx] = data[shuffled_idx, feature_idx]

    elif perturbation_type == 'additive':
        noise = torch.randn_like(perturbed_data[:, feature_idx].float()) * scale * torch.std(perturbed_data[:, feature_idx].float())
        noise = noise.to(perturbed_data.device)

        if data.dtype == torch.int32:
            perturbed_data[non_zero_rows_mask, feature_idx] += torch.tensor(noise[non_zero_rows_mask], dtype=torch.int32).to(perturbed_data.device)
        else:
            perturbed_data[non_zero_rows_mask, feature_idx] += noise[non_zero_rows_mask]

    elif perturbation_type == 'multiplicative':
        factor = 1 + scale * (torch.rand(perturbed_data.shape[0], device=perturbed_data.device) - 0.5)
        if data.dtype == torch.int32:
            perturbed_data[non_zero_rows_mask, feature_idx] = torch.tensor(
                perturbed_data[non_zero_rows_mask, feature_idx].float() * factor[non_zero_rows_mask],
                dtype=torch.int32).to(perturbed_data.device)
        else:
            perturbed_data[non_zero_rows_mask, feature_idx] *= factor[non_zero_rows_mask]

    return perturbed_data, False  # Return perturbed data and flag indicating sufficient samples

def analyze_feature_importance_multi(id, model_config, fold_results, dataset, feature_names, 
            device, analyse_features='all', perturbation_scale=0.1, min_samples_threshold=10, common_samples=True):
    if analyse_features not in ['all', 'RNA', 'ATAC', 'Flux']:
        raise ValueError("analyse_features must be one of 'all', 'RNA', 'ATAC', 'Flux'")
    
    models = []
    for fold in fold_results:
        model_path = fold['best_model_path']
        if id == 'Multi':
            model = create_multimodal_model(model_config, device, use_mlm=False)
        else:
            model = SingleTransformer(id=id, **model_config).to(device)
        # Load weights to CPU first, then move to target device (handles CUDA->MPS/CPU transfer)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    
    # Compute the original latent space once using the cached models
    original_latent, _, _ = get_latent_space_cached(models, fold_results, dataset, device, batch_size=64, common_samples=common_samples)
    
    feature_shifts = []
    skipped_features = []  # Track features skipped due to insufficient samples
    # Unpack multi-modal data
    X, b, y = (dataset.rna_data, dataset.atac_data, dataset.flux_data), dataset.batch_no, dataset.labels
    rna_input, atac_input, flux_input = X[0], X[1], X[2]
    atac_start = rna_input.shape[1] + 1
    flux_start = atac_start + atac_input.shape[1] + 1
    print("atac start", atac_start, "flux start", flux_start)
    perturb_type = 'shuffle'
    if analyse_features in ['RNA', 'all']:
        print("Analyzing RNA features")
        print("Permuting RNA features with", perturb_type)
        for i in tqdm(range(rna_input.shape[1])):
            # Choose perturbation type based on the mean value
             #if rna_input[:, i].float().mean() < 10 else 'multiplicative'
            perturbed_rna, insufficient_samples = perturb_feature(rna_input, i, perturb_type, scale=perturbation_scale, min_samples_threshold=min_samples_threshold)
            if insufficient_samples:
                skipped_features.append((feature_names[i], "RNA", (rna_input[:, i] != 0).sum().item()))
                feature_shifts.append((feature_names[i], 0.0))  # Add with 0 importance
            else:
                perturbed_dataset = MultiModalDataset((perturbed_rna, atac_input, flux_input), b, y)
                perturbed_latent, _, _ = get_latent_space_cached(models, fold_results, perturbed_dataset, device, batch_size=64, common_samples=common_samples)
                shift = measure_shift(original_latent, perturbed_latent)
                feature_shifts.append((feature_names[i], shift))

    if analyse_features in ['ATAC', 'all']:
        print("Analyzing ATAC features")
        print("Permuting ATAC features with", perturb_type)
        for i in tqdm(range(atac_input.shape[1])):
            perturbed_atac, insufficient_samples = perturb_feature(atac_input, i, perturb_type, perturbation_scale, min_samples_threshold=min_samples_threshold)
            if insufficient_samples:
                skipped_features.append((feature_names[atac_start + i], "ATAC", (atac_input[:, i] != 0).sum().item()))
                feature_shifts.append((feature_names[atac_start + i], 0.0))  # Add with 0 importance
            else:
                perturbed_dataset = MultiModalDataset((rna_input, perturbed_atac, flux_input), b, y)
                perturbed_latent, _, _ = get_latent_space_cached(models, fold_results, perturbed_dataset, device, batch_size=64, common_samples=common_samples)
                shift = measure_shift(original_latent, perturbed_latent)
                feature_shifts.append((feature_names[atac_start + i], shift))
            
    if analyse_features in ['Flux', 'all']:
        print("Permuting Flux features with", perturb_type)
        print("Analyzing Flux features")
        for i in tqdm(range(flux_input.shape[1])):
            perturbed_flux, insufficient_samples = perturb_feature(flux_input, i, 'shuffle_all', perturbation_scale, min_samples_threshold=min_samples_threshold)
            if insufficient_samples:
                skipped_features.append((feature_names[flux_start + i], "Flux", (flux_input[:, i] != 0).sum().item()))
                feature_shifts.append((feature_names[flux_start + i], 0.0))  # Add with 0 importance
            else:
                perturbed_dataset = MultiModalDataset((rna_input, atac_input, perturbed_flux), b, y)
                perturbed_latent, _, _ = get_latent_space_cached(models, fold_results, perturbed_dataset, device, batch_size=64, common_samples=common_samples)
                shift = measure_shift(original_latent, perturbed_latent)
                feature_shifts.append((feature_names[flux_start + i], shift))
    
    # Log skipped features
    if skipped_features:
        print(f"\nSkipped {len(skipped_features)} features due to insufficient samples (< {min_samples_threshold}):")
        for feature_name, modality, sample_count in skipped_features:
            print(f"  {feature_name} ({modality}): {sample_count} samples")
    
    return sorted(feature_shifts, key=lambda x: x[1], reverse=True)
