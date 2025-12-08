import os
import anndata as ad
import pandas as pd
from sklearn.preprocessing import StandardScaler
from . import preprocess_data

def load_clones(data_path):
    df_clone = pd.read_csv(data_path, index_col=["cell.bc"])
    df_clone = df_clone[["assay", 'state/fate', 'cell_type', 
                         'most_dominant_fate', 'most_dominant_fate_pct', 
                         "clone_id", "clone.size (RNA & ATAC)", 'clone.size (RNA)', 'clone.size (ATAC)', 
                         '# of D3 cells (RNA)', '# of D3 cells (ATAC)']]
    df_clone.rename({"clone.size (RNA & ATAC)": "clone_size",
                        "clone.size (RNA)": "cells_RNA",
                        'clone.size (ATAC)': "cells_ATAC",
                        '# of D3 cells (ATAC)' : "cells_ATAC_D3",
                        '# of D3 cells (RNA)' : "cells_RNA_D3",
                        'most_dominant_fate': 'label',
                        'most_dominant_fate_pct': 'pct',
                        'state/fate': 'day3_day21'}, inplace=True, axis=1)
    return df_clone

def add_clone_info(adata, clone_path, split=False):
    """
    Adds clone information to the given AnnData object.
    Parameters:
    adata (AnnData): The AnnData object to which clone information will be added.
    clone_path (str): The file path to the clone data.
    split (bool): Whether to split the data into labelled and unlabelled. Default is False.
    Returns:
    AnnData: The modified AnnData object with clone information added.
    """

    df_clone = load_clones(clone_path)
    filtered_obs = adata.obs.join(df_clone, how='inner')

    if split:
        filtered_obs = filtered_obs[(filtered_obs.label=='reprogramming') | (filtered_obs.label=='dead-end')]
        adata_labelled = adata[filtered_obs.index].copy()
        adata_labelled.obs = filtered_obs
        adata_unlabelled = adata[~adata.obs.index.isin(adata_labelled.obs.index)].copy()
        return adata_labelled, adata_unlabelled

    adata = adata[filtered_obs.index]
    adata.obs = filtered_obs
    return adata

def load_rna(data_path, return_raw=True, clone_info=False, clone_path=None):
    """
    Load RNA data from a given file path.
    Parameters:
    - data_path (str): The file path to the RNA data.
    - return_raw (bool): Whether to return the raw counts or not. Default is False.
    - add_clone_info (bool): Whether to add clone information or not. Default is True.
    - clone_path (str): The file path to the clone information. Required if add_clone_info is True.
    Returns:
    - adata_RNA (AnnData): Annotated data object containing the loaded RNA data.
    """

    # Load RNA data
    adata_RNA = ad.read_h5ad(data_path)
    adata_RNA.obs.index = adata_RNA.obs.index.str.replace('_', '-')
    
    # Restore raw counts if necessary
    if return_raw:
        adata_RNA.X = adata_RNA.raw.X.copy() # Copy raw counts to the expression matrix
     
    # Add batch information
    adata_RNA.obs['batch_no'] = adata_RNA.obs.index.to_series().apply(lambda idx: 1 if 'r1' in idx else (2 if 'r2' in idx else 0))
    
    # Add clone information
    if clone_info:
        if clone_path is None:
            raise ValueError("clone_path must be provided if add_clone_info is True.")
        else:
            adata_RNA = add_clone_info(adata_RNA, clone_path)
    
    # Remove unwanted columns
    columns_to_remove = ['orig.ident', 'old_ident', 'cc_score_diff', 'snn_res_0_8', 
                        'seurat_clusters', 
                        'predicted__cca_co_id', 'prediction_score_fib_1', 'prediction_score_fib_0', 
                        'prediction_score_fib_2', 
                        'prediction_score_early_0', 'prediction_score_transition_0', 
                        'prediction_score_transition_1', 
                        'prediction_score_early_1', 'prediction_score_early_2', 'prediction_score_iep_1', 
                        'prediction_score_transition_2', 'prediction_score_iep_2', 'prediction_score_dead_end_1', 
                        'prediction_score_dead_end_0', 'prediction_score_iep_0', 'prediction_score_dead_end_2', 
                        'prediction_score_max', 'snn_res_0_2', 'cellranger_ident', 'metadata_fate_coarse_rev1', 
                        'md_fate_rev1', 'md_fate_coarse_rev1', 'metadata_fate_rev1', 'day3_day21', 'sample_id',
                        'replicate_id', 'cell_type', 'assay']
    intersection = set(columns_to_remove).intersection(adata_RNA.obs.columns)
    if intersection:
        adata_RNA.obs.drop(intersection, axis=1, inplace=True)

    # Rename columns
    columns_to_rename = {'S.Score': 'S_score', 
                         'G2M.Score': 'G2M_score',
                         'nCount_RNA': 'total_counts',
                         'nFeature_RNA': 'n_genes_by_counts',
                         'Phase': 'phase',
                         'percent.mt': 'pct_counts_mt',
                         }
    intersection = set(columns_to_rename.keys()).intersection(adata_RNA.obs.columns)
    if intersection:
        adata_RNA.obs.rename(columns=columns_to_rename, inplace=True)

    return adata_RNA

def load_atac(data_path, clone_info=False, clone_path=None):
    """
    Load ATAC data from a given file path.
    Parameters:
    - data_path (str): The file path to the ATAC data.
    - clone_info (bool): Whether to add clone information or not. Default is False.
    - clone_path (str): The file path to the clone information. Required if add_clone_info is True.
    Returns:
    - adata_atac (AnnData): Annotated data object containing the loaded ATAC data.
    """
    adata_atac = ad.read_h5ad(data_path)
    adata_atac = adata_atac[:,adata_atac.var['name'] != "Crebzf_122"]
    adata_atac.obs.index = adata_atac.obs.index.str.replace('_', '-')

    adata_atac = adata_atac.copy()
    adata_atac.obs['batch_no'] = adata_atac.obs.index.to_series().apply(lambda idx: 1 if 'r1' in idx else (2 if 'r2' in idx else 0))

    columns_to_remove = ['BlacklistRatio', 'CellNames', 'DoubletEnrichment', 
                     'DoubletScore', 'NucleosomeRatio', 'PassQC', 'PromoterRatio', 
                     'ReadsInBlacklist', 'ReadsInPromoter', 'ReadsInTSS', 'TSSEnrichment',
                     'nDiFrags', 'nFrags', 'nMonoFrags', 'nMultiFrags', 
                     'origin']

    intersection = set(columns_to_remove).intersection(adata_atac.obs.columns)
    if intersection:
        adata_atac.obs.drop(intersection, axis=1, inplace=True)

    if clone_info:
        if clone_path is None:
            raise ValueError("clone_path must be provided if add_clone_info is True.")
        else:
            adata_atac_labelled, adata_atac_unlabelled = add_clone_info(adata_atac, clone_path, split=True)
            return adata_atac_labelled, adata_atac_unlabelled
    else:
        # warning that without clone info, the data will be returned as a single object
        print("Warning: Clone information not provided. Returning a single object.")

    return adata_atac

def concat_fluxes(directory, prefix):
    df_list = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, index_col=0)
            df_list.append(df)
    
    if df_list:
        concatenated_df = pd.concat(df_list, axis=0)
    else:
        concatenated_df = pd.DataFrame() 

    return concatenated_df

def load_flux(data_path, prefix='flux_un', clone_info=False, clone_path=None, scale=True, flux_metadata_path=None):
    """
    Load Flux data from a given file path.
    Parameters:
    - data_path (str): The file path to the Flux data.
    - prefix (str): The prefix of the Flux files. Default is 'flux_un'.
    - clone_info (bool): Whether to add clone information or not. Default is False.
    - clone_path (str): The file path to the clone information. Required if add_clone_info is True.
    Returns:
    - adata_Flux_labelled (pd.DataFrame): Annotated data object containing the labelled Flux data.
    - adata_Flux_unlabelled (pd.DataFrame): Annotated data object containing the unlabelled Flux data.
    - bi_labelled (list): List of binary labels for the labelled Flux data.
    - bi_unlabelled (list): List of binary labels for the unlabelled Flux data.
    - labels (list): List of labels for the labelled Flux data.
    """

    adata_Flux_labelled = pd.read_csv(data_path, index_col=0)
    directory = os.path.dirname(data_path)  
    adata_Flux_unlabelled = concat_fluxes(directory, prefix)
    
    adata_Flux_labelled.index = adata_Flux_labelled.index.str.replace('_', '-')
    # adata_Flux_unlabelled.index = adata_Flux_unlabelled.index.str.replace('_', '-')
    
    if scale:
        std_sc = StandardScaler()
        scaled_unl = std_sc.fit_transform(adata_Flux_unlabelled.values)
        scaled_unl += abs(scaled_unl.min())
        adata_Flux_unlabelled = pd.DataFrame(scaled_unl, 
                                                    index=adata_Flux_unlabelled.index, 
                                                    columns=adata_Flux_unlabelled.columns)
        scaled_la = std_sc.transform(adata_Flux_labelled.values)
        scaled_la += abs(scaled_la.min())
        adata_Flux_labelled = pd.DataFrame(scaled_la, 
                                            index=adata_Flux_labelled.index, 
                                            columns=adata_Flux_labelled.columns)
    if flux_metadata_path is not None:
        md = pd.read_csv(flux_metadata_path)[['X', 'rxnName']]
    else:
        md = pd.read_csv("data/datasets/flux/metabolic_model_metadata.csv")[['X', 'rxnName']]
    dict_rename = {}
    for col in adata_Flux_labelled.columns:
        reaction = md[md['X'] == col]['rxnName'].str.replace(" -> ", "→").values
        dict_rename[col] = reaction[0]
    adata_Flux_labelled = adata_Flux_labelled.rename(columns=dict_rename)
    adata_Flux_unlabelled = adata_Flux_unlabelled.rename(columns=dict_rename)    


    if clone_info:
        if clone_path is None:
            raise ValueError("clone_path must be provided if add_clone_info is True.")
        else:
            df_clone = load_clones(clone_path)
            filtered_obs = adata_Flux_labelled.join(df_clone, how='inner')
            labels = filtered_obs['label']
            pcts = filtered_obs['pct']
            bi_labelled = adata_Flux_labelled.index.map(lambda x: 2 if 'r2' in x else 1 if 'r1' in x else 0)
            bi_unlabelled = adata_Flux_unlabelled.index.map(lambda x: 2 if 'r2' in x else 1 if 'r1' in x else 0)   
            adata_Flux_labelled = adata_Flux_labelled.loc[filtered_obs.index]
            return  adata_Flux_labelled, adata_Flux_unlabelled, bi_labelled, bi_unlabelled, labels, pcts
    else:
        print("Warning: Clone information not provided. Returning raw data.")
        return adata_Flux_labelled, adata_Flux_unlabelled


def load_processed_rna(verbose=True, return_raw=True, return_all_features=False):

    if verbose:
         print('Loading RNA data...')
    # Load RNA data labelled
    adata_RNA_labelled = load_rna("data/datasets/rna/all_rna_d3_labelled.h5ad",
                                    return_raw=True,
                                    clone_info=True,
                                    clone_path="data/datasets/clone/clones.csv")
    # Load RNA data unlabelled
    adata_RNA_unlabelled = load_rna("data/datasets/rna/all_rna_d3_unlabelled.h5ad", 
                                    return_raw=True, 
                                    clone_info=False)
   
    if verbose:
        print('Filtering RNA data...')
    adata_RNA_labelled = preprocess_data.filter_rna_cells_genes(adata_RNA_labelled.copy())
    adata_RNA_unlabelled = preprocess_data.filter_rna_cells_genes(adata_RNA_unlabelled.copy())

    if verbose:
        print('Feature Selection by DEGs...')
    deg_list = preprocess_data.get_degs(adata_RNA_labelled, method='t-test')
 
    if verbose:
        print('Filtering Genes...')
    genes_intersection = set(adata_RNA_labelled.var_names).intersection(set(adata_RNA_unlabelled.var_names)).intersection(set(deg_list.gene))
    adata_RNA_labelled_all = adata_RNA_labelled.copy()
    adata_RNA_labelled = adata_RNA_labelled[:, list(genes_intersection)]
    adata_RNA_unlabelled = adata_RNA_unlabelled[:, list(genes_intersection)]
    

    if return_raw:
        gene_indices = [adata_RNA_labelled.raw.var_names.get_loc(gene) for gene in adata_RNA_labelled.var_names]
        adata_RNA_labelled.X = adata_RNA_labelled.raw.X[:, gene_indices].toarray().copy()
        adata_RNA_unlabelled.X = adata_RNA_unlabelled.raw.X[:, gene_indices].copy()

    if return_all_features:
        return adata_RNA_labelled, adata_RNA_unlabelled, deg_list, adata_RNA_labelled_all
    return adata_RNA_labelled, adata_RNA_unlabelled, deg_list

if __name__ == '__main__':
    adata_ATAC_labelled, adata_ATAC_unlabelled = load_atac("data/datasets/atac/all_atac_d3_motif.h5ad", 
                                                          clone_info=True, 
                                                          clone_path="data/datasets/clone/clones.csv")
    print(adata_ATAC_labelled.obs.columns, adata_ATAC_labelled.obs.shape, adata_ATAC_labelled.obs.index[:10])
    print(adata_ATAC_unlabelled.obs.columns, adata_ATAC_unlabelled.obs.shape, adata_ATAC_unlabelled.obs.index[:10])
    print("Data loaded successfully!")




