import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
from models import SingleTransformer
from utils.helpers import create_multimodal_model

def compute_confusion_matrices(id, model_config, fold_results, dataset, device):
    """
    Get confusion matrices for each fold and aggregate them.
    Args:
        id (str): Model ID.
        model_config (dict): Model configuration.
        fold_results (list): List of dictionaries containing fold results.
        cls_valid_loader (torch.utils.data.DataLoader): Validation data loader.
        device (str): Device to use.
    Returns:
        list: List of confusion matrices for each fold and the aggregated confusion
            matrix.
    """
    if id not in ['RNA', 'ATAC', 'Flux', 'Multi']:
            raise ValueError("id must be one of 'RNA', 'ATAC', 'Flux', 'Multi'")
    # Initialize an empty confusion matrix for aggregation
    agg_cm = np.zeros((2, 2), dtype=int)
    cms = []

    for i, fold in enumerate(fold_results, 1):
        model_path = fold['best_model_path']
        state_dict = torch.load(model_path)
        val_subset = Subset(dataset, fold['val_idx'])
        cls_valid_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        if id=='Multi':
            model = create_multimodal_model(model_config, device, use_mlm=False)
        else:
            model = SingleTransformer(id, **model_config).to(device)
        
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        val_preds, val_labels = [], []
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
                preds = preds.cpu().numpy()
                val_preds.append(preds)
                val_labels.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds).ravel()
        val_labels = np.concatenate(val_labels).ravel()
        
        binary_preds = (val_preds >= 0.5).astype(int)
        # print(f"Fold {i} Confusion Matrix:", val_preds)
        cm = confusion_matrix(val_labels, binary_preds)
        agg_cm += cm
        cms.append(cm)

    cms.append(agg_cm)
    return cms
