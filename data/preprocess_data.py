import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def filter_rna_cells_genes(adata, min_genes=100, min_cells=10):
    """
    Filter cells and genes in RNA data.
    Parameters:
    - adata (AnnData): Annotated data object containing the RNA data.
    - min_genes (int): The minimum number of genes to keep a cell. Default is 100.
    - min_cells (int): The minimum number of cells to keep a gene. Default is 10.
    Returns:
    - adata_filtered (AnnData): Annotated data object containing the filtered RNA data.
    """
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata

def get_degs(adata, method='t-test', p_val=0.05, 
             batch_remove=True, batch_key='batch_no', label_key='label', 
             reference='dead-end', target='reprogramming'):
    """
    Get differentially expressed genes (DEGs) from the RNA data.
    """

    sc.pp.normalize_total(adata, target_sum=1e4, exclude_highly_expressed=False)
    sc.pp.log1p(adata)
    if batch_remove:
        sc.pp.combat(adata, key=batch_key)
    
    sc.tl.rank_genes_groups(adata, groupby=label_key, method=method, n_genes=adata.shape[1], use_raw=False, reference=reference)

    de_results = adata.uns['rank_genes_groups']
    gene_list = list(pd.DataFrame(de_results['names'])[target])
    
    # Compute mean and std for each gene in both groups.
    # These are Series indexed by gene names (from adata.var_names).
    group_a_mean_expression = adata[adata.obs[label_key] == reference].to_df().mean()
    group_a_std_expression = adata[adata.obs[label_key] == reference].to_df().std()
    group_b_mean_expression = adata[adata.obs[label_key] == target].to_df().mean()
    group_b_std_expression = adata[adata.obs[label_key] == target].to_df().std()
    
    # Reorder (or reindex) the computed series so that they match the order in gene_list.
    group_a_mean_expression = group_a_mean_expression.reindex(gene_list)
    group_a_std_expression = group_a_std_expression.reindex(gene_list)
    group_b_mean_expression = group_b_mean_expression.reindex(gene_list)
    group_b_std_expression = group_b_std_expression.reindex(gene_list)

    # Create the DEG DataFrame.
    df = pd.DataFrame({
        'gene': gene_list,
        'mean_exp_de': group_a_mean_expression.values,  # 'dead-end' (reference)
        'mean_exp_re': group_b_mean_expression.values,  # 'reprogramming' (target)
        'std_exp_de': group_a_std_expression.values,
        'std_exp_re': group_b_std_expression.values,    
        'pval': de_results['pvals'][target],
        'pval_adj': de_results['pvals_adj'][target],
        'log_fc': de_results['logfoldchanges'][target],
    })

    df['group'] = df.apply(lambda row: reference if row['log_fc'] < 0 else target, axis=1)

    df.sort_values(by='pval_adj', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['pval_adj_log'] = -np.log10(df['pval_adj'])

    df = df[(df.pval_adj < p_val) & ((df.log_fc < -1) | ((df.log_fc > 1) & (df.log_fc < 7)))]
    return df

def get_flux_degs(adata_Flux_labelled, labels):
    dead_end = adata_Flux_labelled[labels.values == "dead-end"]
    reprogramming = adata_Flux_labelled[labels.values == "reprogramming"]

    features = []
    log_fold_changes = []
    p_values = []
    mean_des = []
    mean_res = []
    std_des = []
    std_res = []

    for feature in adata_Flux_labelled.columns:
        mean_de = dead_end[feature].mean()
        mean_re = reprogramming[feature].mean()
        std_de = dead_end[feature].std()
        std_re = reprogramming[feature].std()
        
        log_fold_change = np.log2(mean_re + 1e-10) - np.log2(mean_de + 1e-10)
        t_stat, p_value = ttest_ind(dead_end[feature], reprogramming[feature], nan_policy="omit")
        mean_des.append(mean_de)
        mean_res.append(mean_re)
        std_des.append(std_de)
        std_res.append(std_re)
        features.append(feature)
        log_fold_changes.append(log_fold_change)
        p_values.append(p_value)

    adjusted_p_values = multipletests(p_values, method="fdr_bh")[1]

    df_flux_degs = pd.DataFrame({
        "feature": features,
        "mean_de": mean_des,
        "mean_re": mean_res,
        "mean_diff": np.array(mean_res) - np.array(mean_des),
        "std_de": std_des,
        "std_re": std_res,
        "log_fc": log_fold_changes,
        "pval": p_values,
        "pval_adj": adjusted_p_values,
        'pval_adj_log' : -np.log10(adjusted_p_values)
    })
    df_flux_degs['group'] = df_flux_degs.apply(lambda row: 'dead-end' if row['mean_de'] > row['mean_re'] else 'reprogramming', axis=1)
    df_flux_degs = df_flux_degs.sort_values(by="pval_adj").reset_index(drop=True)
    return df_flux_degs

def get_atac_degs(adata, method='t-test', label_key='label', 
             reference='dead-end', target='reprogramming'):
    """
    Get differentially expressed genes (DEGs) from the ATAC data.
    """
    
    sc.tl.rank_genes_groups(adata, groupby=label_key, method=method, 
                            n_genes=adata.shape[1], use_raw=False, reference=reference)

    group_a_mean_expression = adata[adata.obs[label_key] == reference].to_df().mean()
    group_a_std_expression = adata[adata.obs[label_key] == reference].to_df().std()
    group_b_mean_expression = adata[adata.obs[label_key] == target].to_df().mean()
    group_b_std_expression = adata[adata.obs[label_key] == target].to_df().std()
    de_results = adata.uns['rank_genes_groups']
    features = list(pd.DataFrame(de_results['names'])[target])

    # Reindex the mean and std Series to this feature list
    mean_de = group_a_mean_expression.reindex(features)
    mean_re = group_b_mean_expression.reindex(features)
    std_de = group_a_std_expression.reindex(features)
    std_re = group_b_std_expression.reindex(features)

    min_val = min(mean_de.min(), mean_re.min())
    # Determine a shift value so that the smallest value becomes a small positive number.
    shift = 0
    if min_val <= 0:
        shift = abs(min_val) + 1e-10
    df = pd.DataFrame({
        'feature': list(pd.DataFrame(de_results['names'])[target]),
        'pval': de_results['pvals'][target],
        'pval_adj': de_results['pvals_adj'][target],
        'log_fc': np.log2(mean_re + shift) - np.log2(mean_de + shift),
        'mean_de': mean_de,
        'mean_re': mean_re,
        'mean_diff': mean_re - mean_de,
        'std_de': std_de,
        'std_re': std_re,    

    })

    df['group'] = df.apply(lambda row: 'dead-end' if row['mean_de'] > row['mean_re'] else 'reprogramming', axis=1)

    df.sort_values(by='pval_adj', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['pval_adj_log'] = -np.log10(df['pval_adj'])
    return df
