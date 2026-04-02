import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from utils.helpers import create_multimodal_model
from models import SingleTransformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

def filter_idx(dataset, idx):
    """
    Filter the idx to only return the samples that none of its modalities are all zeros
    Args:
        dataset: Dataset object containing the data.
        idx: List of indices to filter.
    Returns:
        filtered_idx: List of filtered indices.
    """
    rna = dataset.rna_data
    atac = dataset.atac_data
    flux = dataset.flux_data
    mask = (rna != 0).any(axis=1) & (atac != 0).any(axis=1) & (flux != 0).any(axis=1)
    # filter the idx if the id is in the mask
    filtered_idx = [i for i in idx if mask[i]]
    
    return filtered_idx

def analyze_cls_attention(id, fold_results, dataset, model_config, device, indices, 
                          average_heads=True, return_flow_attention=False):
    """
    Extracts the attention weights of the validation set of each fold
    Args:
        id: The type of data to use. Must be one of 'RNA', 'ATAC', 'Flux', 'Multi'.
        fold_results: List of dictionaries containing the results of each fold.
        dataset: Dataset object containing the data.
        model_config: Dictionary containing the model configuration.
        device: Device to run the model on.
        sample_type: The type of samples to analyze. Must be one of 'all', 'dead-end', or 'reprogramming'. Defaults to 'all'.
        average_heads: Whether to average the attention weights across heads. Defaults to True.
    Returns:
        all_attention_weights: Numpy array containing the attention weights of the validation set
    """
    if id not in ['RNA', 'ATAC', 'Flux', 'Multi']:
        raise ValueError("id must be one of 'RNA', 'ATAC', 'Flux', 'Multi'")
    
    all_attention_weights = []

    for fold in fold_results:
        
        val_idx = fold['val_idx']
        # filter val_idx if is in indices
        val_idx = [i for i in val_idx if i in indices]
        
        if id == 'Multi':
            val_idx = filter_idx(dataset, val_idx)

        if len(val_idx) == 0:
            print('No samples of the specified type in the validation set. Skipping...')
            continue

        val_ds = Subset(dataset, val_idx)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        if id=='Multi':
            model = create_multimodal_model(model_config, device, use_mlm=False)
        else:
            model = SingleTransformer(id=id, **model_config).to(device)

        model_path = fold['best_model_path']
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        with torch.no_grad():
            for batch in val_loader:
                x, b, _ = batch
                if isinstance(x, list):
                    rna = x[0].to(device)
                    atac = x[1].to(device)
                    flux = x[2].to(device)
                    x = (rna, atac, flux)
                else:
                    x = x.to(device)
                b = b.to(device)
                
                _, _, attention_weights = model(x, b, return_attention=True, return_flow_attention=return_flow_attention)
                
                if not return_flow_attention:
                    if average_heads:
                        attention_weights = attention_weights.squeeze(-2).mean(dim=1)  # Average across heads (batch, 1, seq_len) -> (batch, seq_len)
                    else:
                        attention_weights = attention_weights.squeeze(-2)  # (batch, num_heads, 1, seq_len) -> (batch, num_heads, seq_len)
                
                # if hasattr(attention_weights, 'numpy'):
                #     attention_weights = attention_weights.cpu().numpy()
                all_attention_weights.append(attention_weights)

    if not return_flow_attention:
        return np.concatenate(all_attention_weights, axis=0) # (n_samples, seq_len) or (n_samples, num_heads, seq_len)
    else:
        att_w = {'rna': [], 'atac': [], 'flux': [], 'cls': []}
        # noew we have a dict. So concatenating all values for each key
        num_layers_mlm = len(all_attention_weights[0]['rna'])
        num_layers_cls = len(all_attention_weights[0]['cls']) if isinstance(all_attention_weights[0]['cls'], list) else 1

        for key in all_attention_weights[0].keys():
            key_all_attentions = []
            for batch_row in all_attention_weights:
                modality_batch_attention_layers = batch_row[key]
                if isinstance(modality_batch_attention_layers, list):
                    for i, modality_attention_layers in enumerate(modality_batch_attention_layers):
                        modality_batch_attention_layers[i] = modality_attention_layers.cpu()
                    key_all_attentions.append(modality_batch_attention_layers)
                else:
                    key_all_attentions.append([modality_batch_attention_layers.cpu()])
            # now I have a list of attention weights for each batch in each layer [[layer0_att_weights_batch1, layer1_att_weights_batch1, ...], [layer0_att_weights_batch2, layer1_att_weights_batch2, ...], ...]
            # I want to concatenate all the attention weights for each layer
            num_layers = num_layers_cls if key == 'cls' else num_layers_mlm
            att_w[key] = [torch.cat([layer[i] for layer in key_all_attentions], axis=0) for i in range(num_layers)]
    return att_w

def compute_attention_rollout(attention_weights):
    """
    Computes the attention rollout for a batch of samples.
    Expects attention_weights to be a list (length=num_layers) of tensors 
    with shape (batch, num_heads, seq_len, seq_len). For each layer, we average 
    over the heads and then compute the rollout per sample.

    Returns:
        rollout: A tensor of shape (batch, seq_len, seq_len) representing the 
                 effective attention from the input token (typically CLS) to all tokens.
    """
    num_layers = len(attention_weights)
    # Get batch size and sequence length from the first layer's tensor
    batch_size, num_heads, seq_len, _ = attention_weights[0].shape

    # Initialize the combined attention as the identity matrix for each sample
    combined_attention = torch.eye(seq_len, device=attention_weights[0].device)
    combined_attention = combined_attention.unsqueeze(0).repeat(batch_size, 1, 1)

    for layer in range(num_layers):
        # Average over heads to get a (batch, seq_len, seq_len) tensor for this layer
        layer_attention = attention_weights[layer].mean(dim=1)
        # Update the rollout for each sample using batched matrix multiplication
        combined_attention = torch.bmm(layer_attention, combined_attention)
    return combined_attention

def multimodal_attention_rollout(all_attention_weights):
    rna_rollout = compute_attention_rollout(all_attention_weights['rna'])
    atac_rollout = compute_attention_rollout(all_attention_weights['atac'])
    flux_rollout = compute_attention_rollout(all_attention_weights['flux'])
    
    cls_attention = all_attention_weights['cls'][0].mean(dim=1).squeeze(1)   # Average over heads
    
    # Split CLS attention for each modality
    rna_cls_attn, atac_cls_attn, flux_cls_attn = cls_attention.split(
        [rna_rollout.size(1), atac_rollout.size(1), flux_rollout.size(1)], dim=1)
    
    final_rollout = torch.cat([
        rna_cls_attn.unsqueeze(1) @ rna_rollout,
        atac_cls_attn.unsqueeze(1) @ atac_rollout,
        flux_cls_attn.unsqueeze(1) @ flux_rollout
    ], dim=2)
    
    return final_rollout.squeeze(1) # remove head dimension [samples, tokens]

def print_top_features(attention_weights, feature_names, top_n=5, modality=None):
    print(f"\nTop {top_n} attended features ({modality} samples):")
    avg_attention = attention_weights.mean(axis=0).numpy() if hasattr(attention_weights, 'numpy') else attention_weights.mean(axis=0)
    top_indices = avg_attention.argsort()[-top_n:][::-1]
    for i in top_indices:
        print(f"{feature_names[i]}: {avg_attention[i]:.4f}")

def get_top_features(attention_weights, feature_names, top_n=100, modality=None):
    ls = []
    avg_attention = attention_weights.mean(axis=0).numpy() if hasattr(attention_weights, 'numpy') else attention_weights.mean(axis=0)
    if top_n:
        top_indices = avg_attention.argsort()[-top_n:][::-1]
    else:
        top_indices = avg_attention.argsort()[::-1]
        
    for i in top_indices:
        ls.append((feature_names[i],avg_attention[i]))
    return ls

def compute_attention_flow(attention_weights):
    num_layers = len(attention_weights)
    num_tokens = attention_weights[0].size(-1)
    
    # Create adjacency matrix for the flow network
    adj_matrix = np.zeros((num_layers * num_tokens, num_layers * num_tokens))
    
    for i in range(num_layers - 1):
        layer_attention = attention_weights[i].mean(dim=1).cpu().numpy()  # Average over heads
        start_idx = i * num_tokens
        end_idx = (i + 1) * num_tokens
        adj_matrix[start_idx:end_idx, end_idx:(end_idx + num_tokens)] = layer_attention
    
    for i in range(num_layers - 1):
        start_idx = i * num_tokens
        end_idx = (i + 1) * num_tokens
        adj_matrix[start_idx:end_idx, end_idx:(end_idx + num_tokens)] += np.eye(num_tokens)
    
    flows = np.zeros((num_tokens, num_tokens))
    for i in range(num_tokens):
        source = i
        for j in range(num_tokens):
            sink = (num_layers - 1) * num_tokens + j
            _, flow = maximum_flow(csr_matrix(adj_matrix), source, sink)
            flows[i, j] = flow
    
    return torch.tensor(flows, device=attention_weights[0].device)

def multimodal_attention_flow(all_attention_weights):
    rna_flow = compute_attention_flow(all_attention_weights['rna'])
    atac_flow = compute_attention_flow(all_attention_weights['atac'])
    flux_flow = compute_attention_flow(all_attention_weights['flux'])
    
    cls_attention = all_attention_weights['cls'][0].mean(dim=1).squeeze(1)  # Average over heads
    
    # Split CLS attention for each modality
    rna_cls_attn, atac_cls_attn, flux_cls_attn = cls_attention.split(
        [rna_flow.size(1), atac_flow.size(1), flux_flow.size(1)], dim=1)
    
    # Normalize flows
    rna_flow = rna_flow / rna_flow.sum(dim=1, keepdim=True)
    atac_flow = atac_flow / atac_flow.sum(dim=1, keepdim=True)
    flux_flow = flux_flow / flux_flow.sum(dim=1, keepdim=True)
    
    final_flow = torch.cat([
        rna_cls_attn.unsqueeze(1) @ rna_flow,
        atac_cls_attn.unsqueeze(1) @ atac_flow,
        flux_cls_attn.unsqueeze(1) @ flux_flow
    ], dim=2)
    
    return final_flow.squeeze(1)