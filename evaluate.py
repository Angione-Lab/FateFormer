import numpy as np
import torch 
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from utils.helpers import create_masked_input, create_multimodal_model
from utils.losses import MLMLoss
from models import SingleTransformer

def evaluate_mlm(model, loader, mask_token=-1, mse_based=False, device='cpu'):
    model.eval()
    total_loss = 0
    total_batches = len(loader)
    criterion = MLMLoss(mse_based=mse_based)
    with torch.no_grad():
        for inputs, bi in tqdm(loader, desc="Evaluating MLM"):
            inputs = inputs.to(device)
            bi = bi.to(device)
            
            masked_inputs, mask = create_masked_input(inputs, mask_token)
            masked_inputs = masked_inputs.to(device)
            mask = mask.to(device)
            predictions = model(masked_inputs, bi, masked_lm=True)
            loss = criterion(predictions, inputs, mask)
            total_loss += loss.item()

    avg_loss = total_loss / total_batches
    return avg_loss


def evaluate_cls_cv(id, fold_results, model_config, dataset, device='cpu'):
     """
     Cross Evaluate the classifier on the validation sets.
     Args:
         id (str): Model ID.
         fold_results (list): List of dictionaries containing fold results.
         model_config (dict): Model configuration.
         dataset (torch.utils.data.Dataset): Dataset.
         device (str): Device to use.
     Returns:
         list: List of stored AUCs.
         list: List of AUCs.
     """
     if id not in ['RNA', 'ATAC', 'Flux', 'Multi']:
        raise ValueError("id must be one of 'RNA', 'ATAC', 'Flux', 'Multi'")
    
     aucs_stored, aucs = [], []
     val_preds, val_labels = [], []
     for i, fold in tqdm(enumerate(fold_results, 1), desc="Evaluating Classifier"):
        model_path = fold['best_model_path']
        state_dict = torch.load(model_path)
        val_subset = Subset(dataset, fold['val_idx'])
        cls_valid_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        if id=='Multi':
            model = create_multimodal_model(model_config, device, use_mlm=False)
        else:
            model = SingleTransformer(id, **model_config).to(device)

        model.load_state_dict(state_dict, strict=True)

        val_auc, val_pred, val_label = evaluate_cls(model, cls_valid_loader, device)
        val_auc_saved = fold['best_val_auc']
        aucs_stored.append(val_auc_saved)
        aucs.append(val_auc)
        val_preds.append(val_pred)
        val_labels.append(val_label)

     return aucs_stored, aucs, val_preds, val_labels

def evaluate_cls(model, loader, device):
    """
    Evaluate the classifier on the validation set.
    Args:
        model (torch.nn.Module): Model.
        loader (torch.utils.data.DataLoader): Data loader.
        device (str): Device to use.
    Returns:
        float: AUC score.
        list: List of predictions.
        list: List of labels.
    """
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for inputs, bi, y in loader:
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

    val_auc = roc_auc_score(val_labels, val_preds)
    
    return val_auc, val_preds, val_labels

