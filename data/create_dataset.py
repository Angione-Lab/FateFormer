import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.dataset import Dataset
from anndata import AnnData
import pandas as pd
import random
import numpy as np

def get_mlm_loaders(train_data, val_data, batch_size=32, batch_key='batch_no', data_dtype=torch.float32):
    if isinstance(train_data, AnnData) and \
       isinstance(val_data, AnnData):
        X_train = torch.tensor(train_data.X.toarray().copy(), dtype=data_dtype)
        b_train = torch.tensor(train_data.obs[batch_key], dtype=torch.int32)

        X_val = torch.tensor(val_data.X.toarray().copy(), dtype=data_dtype)
        b_val = torch.tensor(val_data.obs[batch_key], dtype=torch.int32)

    elif isinstance(train_data, tuple) and \
         isinstance(train_data[0], (pd.DataFrame)) and \
         isinstance(val_data, (tuple)) and \
         isinstance(val_data[0], (pd.DataFrame)):
        
        X_train = torch.tensor(train_data[0].values, dtype=data_dtype)
        b_train = torch.tensor(train_data[1], dtype=torch.int32)

        X_val = torch.tensor(val_data[0].values, dtype=data_dtype)
        b_val = torch.tensor(val_data[1], dtype=torch.int32)
    else:
        raise ValueError("Data must be an AnnData object or a tuple of (pd.DataFrame, list).")
    
    mlm_train_dataset = TensorDataset(X_train, b_train)
    mlm_train_loader = DataLoader(mlm_train_dataset, batch_size=batch_size, shuffle=True)

    mlm_val_dataset = TensorDataset(X_val, b_val)
    mlm_val_loader = DataLoader(mlm_val_dataset, batch_size=batch_size, shuffle=False)

    return mlm_train_loader, mlm_val_loader


def get_cls_dataset(data, batch_key='batch_no', label_key='label', 
                    pct_key='pct', filter_pcts=50.0, 
                    data_dtype=torch.float32):

    if isinstance(data, AnnData):
        X = torch.tensor(data.X.toarray().copy(), dtype=data_dtype)
        y = torch.tensor([{'reprogramming':1, 'dead-end':0}[i] for i in list(data.obs[label_key])], dtype=torch.float32)
        b = torch.tensor(data.obs[batch_key], dtype=torch.int32)
        pcts = torch.tensor(data.obs[pct_key], dtype=torch.float32)

        X = X[pcts > filter_pcts]
        y = y[pcts > filter_pcts]
        b = b[pcts > filter_pcts]
        pcts = pcts[pcts > filter_pcts]
        feature_names = data.var_names.tolist()

    elif isinstance(data, tuple) and isinstance(data[0], pd.DataFrame):
        X = torch.tensor(data[0].values, dtype=data_dtype)
        y = torch.tensor([{'reprogramming':1, 'dead-end':0}[i] for i in list(data[1])], dtype=torch.float32)
        b = torch.tensor(data[2], dtype=torch.int32)
        pcts = torch.tensor(data[3], dtype=torch.float32)
        X = X[pcts > filter_pcts]
        y = y[pcts > filter_pcts]
        b = b[pcts > filter_pcts]
        pcts = pcts[pcts > filter_pcts]
        feature_names = data[0].columns.tolist()

    else:
        raise ValueError("Data must be an AnnData object or a tuple of (pd.DataFrame, list, list, list).")
 
    dataset = TensorDataset(X, b, y)
    
    return dataset, pcts, feature_names

def get_pair_modalities(adata_rna, adata_atac, flux_df, include_unused_atacs=False, seed=42):
    """
    Pair RNA, ATAC and Flux data based on clone IDs.
    Args:
        adata_rna (AnnData): RNA data.
        adata_atac (AnnData): ATAC data.
        flux_df (pd.DataFrame): Flux data.
        include_unused_atacs (bool): Include ATAC samples that do not have a paired RNA sample.
    Returns:
        tuple:
         - rna_data (pd.DataFrame): RNA data matched by clone IDs, with rows representing samples and columns representing gene expressions.
         - atac_data (pd.DataFrame): ATAC data matched by clone IDs, with rows representing samples and columns representing chromatin accessibility features.
         - flux_data (pd.DataFrame): Flux data matched by clone IDs, with rows representing samples and columns representing flux measurements.
    
        np.array: labels. np.array of labels.
        np.array: batch indices. np.array of batch indices.
        pd.DataFrame: indices. A DataFrame where each row contains the indices of matched RNA and ATAC samples. 
                                  If no match is found for one modality, the corresponding value is None.
        np.array: pcts. Array of dominant fate percentages for each paired sample.
    """

    # Create a dictionary to map ATAC clone IDs to their indices
    atac_clone_to_indices = {clone_id: [] for clone_id in adata_atac.obs['clone_id'].unique()}
    adata_atac.obs['index'] = adata_atac.obs.index
    grouped = adata_atac.obs.groupby('clone_id')['index'].apply(list)
    atac_clone_to_indices.update(grouped)

    rna_data, atac_data, flux_data, labels, batch_ind, indices, pcts = [], [], [], [], [], [], []
    
    used_atac_indices = set()
    
    for rna_index, row in adata_rna.obs.iterrows():
        clone_id = row['clone_id']
        sibling_atac_indices = [idx for idx in atac_clone_to_indices.get(clone_id, []) if idx not in used_atac_indices]

        if sibling_atac_indices:
            random.seed(seed)
            atac_index = random.choice(sibling_atac_indices)
            # atac_index = sibling_atac_indices[0]
            
            used_atac_indices.add(atac_index)
            
            rna_sample = adata_rna[rna_index].X.toarray().flatten() if hasattr(adata_rna[rna_index].X, 'toarray') else adata_rna[rna_index].X
            atac_sample = adata_atac[atac_index].X.toarray().flatten() if hasattr(adata_atac[atac_index].X, 'toarray') else adata_atac[atac_index].X
        else:
            rna_sample = adata_rna[rna_index].X.toarray().flatten() if hasattr(adata_rna[rna_index].X, 'toarray') else adata_rna[rna_index].X
            atac_sample = np.zeros(adata_atac.shape[1])  # Fill with zeros if no ATAC pair is found

        flux_sample = flux_df.loc[rna_index].values

        label = row['label']
        bt = row['batch_no']
        pct = row['pct']
        
        rna_data.append(rna_sample)
        atac_data.append(atac_sample)
        flux_data.append(flux_sample)
        labels.append(label)
        batch_ind.append(bt)
        pcts.append(pct)
        indices.append((rna_index, atac_index) if sibling_atac_indices else (rna_index, None))
        
    
    if include_unused_atacs:
        all_atac_indices = set(adata_atac.obs.index)
        unused_atac_indices = sorted(list(all_atac_indices - used_atac_indices))
        unused_atac_samples = adata_atac[list(unused_atac_indices)]

        for atac_index in unused_atac_indices:
            atac_sample = unused_atac_samples[atac_index].X.toarray().flatten() if hasattr(unused_atac_samples[atac_index].X, 'toarray') else unused_atac_samples[atac_index].X
            rna_sample = np.zeros(adata_rna.shape[1])  # Fill with zeros for RNA
            flux_sample = np.zeros(flux_df.shape[1])   # Fill with zeros for flux

            label = adata_atac.obs.loc[atac_index, 'label']
            bt = adata_atac.obs.loc[atac_index, 'batch_no']
            pct = adata_atac.obs.loc[atac_index, 'pct']

            rna_data.append(rna_sample)
            atac_data.append(atac_sample)
            flux_data.append(flux_sample)
            labels.append(label)
            batch_ind.append(bt)
            pcts.append(pct)
            indices.append((None, atac_index))
        
    rna_data = pd.DataFrame(rna_data, columns=adata_rna.var_names, index=indices)
    atac_data = pd.DataFrame(atac_data, columns=adata_atac.var_names, index=indices)
    flux_data = pd.DataFrame(flux_data, columns=flux_df.columns, index=indices)
    
    X_i = (rna_data, atac_data, flux_data)
    y_i = np.array(labels)
    b_i = np.array(batch_ind)
    indices = pd.DataFrame(np.array(indices), columns=["RNA", "ATAC"])
    pcts = np.array(pcts)
    
    return X_i, y_i, b_i, indices, pcts

class MultiModalDataset(Dataset):
    """
    Multi-modal dataset for RNA, ATAC, and Flux data.
    Args:
        X (tuple): Tuple of (RNA, ATAC, Flux) data.
        batch_no (list): List of batch indices.
        labels (list): List of labels.
    """
    def __init__(self, X, batch_no, labels, df_indics=None, pcts=None, label_names=None):
        if isinstance(X[0], pd.DataFrame):
            self.rna_data = torch.tensor(X[0].values, dtype=torch.int32)
            self.atac_data = torch.tensor(X[1].values, dtype=torch.float32)
            self.flux_data = torch.tensor(X[2].values, dtype=torch.float32)
        else:
            self.rna_data = torch.tensor(X[0], dtype=torch.int32)
            self.atac_data = torch.tensor(X[1], dtype=torch.float32)
            self.flux_data = torch.tensor(X[2], dtype=torch.float32)

        self.batch_no = torch.tensor(batch_no, dtype=torch.int32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.df_indics = df_indics
        self.pcts = pcts
        self.label_names = label_names
    def __len__(self):
        return len(self.labels)

    def get_df_indices(self):
        return self.df_indics
    def get_pcts(self):
        return self.pcts
    def get_label_names(self):
        return self.label_names
    def __getitem__(self, idx):
        rna_sample = self.rna_data[idx]
        atac_sample = self.atac_data[idx]
        flux_sample = self.flux_data[idx]
        batch_no = self.batch_no[idx]
        label = self.labels[idx]
        return (rna_sample, atac_sample, flux_sample), batch_no, label

