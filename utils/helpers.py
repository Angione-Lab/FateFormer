import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import SingleTransformer, MultiModalTransformer
import config
from data import create_dataset

def create_masked_input(input_tensor, mask_token, mask_prob=0.20):
    """
    Creates a masked input tensor by randomly replacing elements with a mask token.
    Args:
        input_tensor (torch.Tensor): The input tensor to be masked.
        mask_token: The token to be used for masking.
        mask_prob (float, optional): The probability of masking an element. Defaults to 0.20.
    Returns:
        torch.Tensor: The masked input tensor.
        torch.Tensor: A boolean mask indicating which elements were masked.
    """

    mask = torch.rand(input_tensor.shape) < mask_prob
    masked_input = input_tensor.clone()
    masked_input[mask] = mask_token
    return masked_input, mask

def get_max(adata):
    """
    Get the maximum value in the data.
    Args:
        adata (list): A list of AnnData objects.
    Returns:
        float: The maximum value in the list data.
    """
    assert(isinstance(adata, list)), "adata must be a list of AnnData objects."
    x_s = []
    for i in adata:
        X = torch.tensor(i.X.toarray().copy())
        x_s.append(np.array(X).flatten().max())
    return max(x_s)

def get_token_embeddings(model, dataset, device):
    """
    Get the token embeddings for the dataset.
    Args:
        model (torch.nn.Module): Model.
        dataset (torch.utils.data.Dataset): Dataset.
        device (str): Device to use.
    Returns:
        torch.Tensor: Embeddings.
    """
    model.eval()
    embeddings = []
    loader = DataLoader(dataset, batch_size=32, shuffle=False) 
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, bi, _ = batch
            elif len(batch) == 2:
                inputs, bi = batch
            if isinstance(inputs, list):
                rna= inputs[0].to(device)
                atac = inputs[1].to(device)
                flux = inputs[2].to(device)
                inputs = (rna, atac, flux)
            else:
                inputs = inputs.to(device)
            bi = bi.to(device)

            output = model(inputs, bi, return_embeddings=True)
            embeddings.append(output.cpu().detach())
    
    # Concatenate embeddings across batches
    embeddings = torch.cat(embeddings, dim=0)  # shape: (n_samples, seq_len, d_model)
    return embeddings

def get_all_modalities_available_samples(dataset):
    
    rna = dataset.rna_data
    atac = dataset.atac_data
    flux = dataset.flux_data
    mask = (rna != 0).any(axis=1) & (atac != 0).any(axis=1) & (flux != 0).any(axis=1)
    new_ds = create_dataset.MultiModalDataset((rna[mask], atac[mask], flux[mask]), 
                                              dataset.batch_no[mask], 
                                              dataset.labels[mask])
    return new_ds

def separate_dataset(ds):
    """
    Separate a dataset into two groups based on the labels.
    Args:
        ds (TensorDataset): Dataset.
    Returns:
        TensorDataset: Dataset with label 0.
        TensorDataset: Dataset with label 1.
    """
    X, b, y = ds.tensors

    # Create masks for labels 0 and 1
    mask_0 = (y == 0)
    mask_1 = (y == 1)

    # Filter the tensors based on the masks
    X_0, b_0, y_0 = X[mask_0], b[mask_0], y[mask_0]
    X_1, b_1, y_1 = X[mask_1], b[mask_1], y[mask_1]

    # Create new datasets for each group
    dataset_0 = TensorDataset(X_0, b_0, y_0)  # Dataset with y == 0
    dataset_1 = TensorDataset(X_1, b_1, y_1)

    return dataset_0, dataset_1

def create_multimodal_model(model_config, device, use_mlm=False):
    """
    Create a multimodal model.
    Args:
        model_config (dict): Model configuration.
        device (str): Device to use.
        use_mlm (bool, optional): Whether to use MLM pretraining. Defaults to False.
    Returns:
        MultiModalTransformer: Multimodal model.
    """
    model_config_rna, model_config_atac, model_config_flux = model_config['RNA'], model_config['ATAC'], model_config['Flux']
    share_config, model_config_multi = model_config['Share'], model_config['Multi']
    rna_model = SingleTransformer("RNA", **model_config_rna, **share_config).to(device)
    atac_model = SingleTransformer("ATAC", **model_config_atac, **share_config).to(device)
    flux_model = SingleTransformer("Flux", **model_config_flux, **share_config).to(device)
    if use_mlm:
            rna_model.load_state_dict(torch.load(config.MLM_RNA_CKP), strict=False)
            atac_model.load_state_dict(torch.load(config.MLM_ATAC_CKP), strict=False)
            flux_model.load_state_dict(torch.load(config.MLM_FLUX_CKP), strict=False)
            # print("Loaded MLM pretraining weights.: \n RNA: {}, ATAC: {}, Flux: {}".format(config.MLM_RNA_CKP, config.MLM_ATAC_CKP, config.MLM_FLUX_CKP))
    model = MultiModalTransformer(rna_model, atac_model, flux_model, **model_config_multi).to(device)
    return model