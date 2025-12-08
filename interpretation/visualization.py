
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats
from scipy.stats import ttest_rel
import pandas as pd

def plot_conf_matrix_mlm_vs_nomlm(cms_mlm, cms_nomlm, m_type, only_agg=True, suptitle="Confusion Matrix Comparison"):

    labels = ['Dead-end', 'Reprogramming']
    
    if only_agg:
        # Plot only the aggregated confusion matrices (last one in each list)
        cms_mlm_agg = cms_mlm[-1]
        cms_nomlm_agg = cms_nomlm[-1]

        f = plt.figure(figsize=(12, 5))  
        plt.suptitle(suptitle, fontsize=16)

        # Plot confusion matrix for aggregated MLM
        plt.subplot(1, 2, 1)
        sns.heatmap(cms_mlm_agg, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - MLM (Aggregated)')

        # Plot confusion matrix for aggregated No MLM
        plt.subplot(1, 2, 2)
        sns.heatmap(cms_nomlm_agg, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - No MLM (Aggregated)')

        f.savefig(f'./figures/confusion_matrices_{m_type}.pdf', bbox_inches='tight')
        plt.tight_layout() 
        plt.show()
    
    else:
        # Plot confusion matrices for each fold
        n_folds = len(cms_mlm)
        f = plt.figure(figsize=(15, 2 * n_folds))  # Adjust figure size according to the number of folds
        plt.suptitle(suptitle, fontsize=16)

        for i in range(n_folds):
            # Plot confusion matrix for MLM in the first row (subplot)
            plt.subplot(n_folds, 2, i*2 + 1)  # First column (MLM)
            sns.heatmap(cms_mlm[i], annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - MLM (Fold {i+1})')

            # Plot confusion matrix for No MLM in the second column (subplot)
            plt.subplot(n_folds, 2, i*2 + 2)  # Second column (No MLM)
            sns.heatmap(cms_nomlm[i], annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - No MLM (Fold {i+1})')

        f.savefig(f'./figures/confusion_matrices_folds_{m_type}.pdf', bbox_inches='tight')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.show()

def plot_training_vs_validation_losses(train_losses, val_losses, title="Losses"):
    epochs = len(train_losses)
    f = plt.figure(figsize=(10, 3))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss')

    f.savefig('./figures/losses.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_roc_auc_curve(val_preds, val_labels, m_type, aggregate=False):
    
    if aggregate:
        # Aggregate all folds into one list
        all_labels = np.concatenate(val_labels).ravel()
        all_preds = np.concatenate(val_preds).ravel()
        auc = roc_auc_score(all_labels, all_preds)
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        
        f = plt.figure()
        plt.plot(fpr, tpr, label=f'Aggregated AUC: {auc:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Aggregated)')
        plt.legend()
        f.savefig(f'./figures/roc_curve_{m_type}.pdf', bbox_inches='tight')
        plt.show()
    
    else:
        # Plot AUC for each fold separately
        f = plt.figure()
        for i, (labels, preds) in enumerate(zip(val_labels, val_preds), 1):
            auc = roc_auc_score(labels, preds)
            fpr, tpr, _ = roc_curve(labels, preds)
            
            plt.plot(fpr, tpr, label=f'Fold {i} AUC: {auc:.4f}')
        
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Each Fold)')
        plt.legend()
        f.savefig(f'./figures/roc_curve_{m_type}.pdf', bbox_inches='tight')
        plt.show()


def plot_auc_boxplot_comparison(fold_results1, fold_results2, title="AUC Comparison"):
    """Plot AUC box comparison between two models."""

    train_auc_scores_mlm = [fold['train_auc'] for fold in fold_results1]
    train_auc_scores_nomlm = [fold['train_auc'] for fold in fold_results2]
    val_auc_scores_mlm = [fold['best_val_auc'] for fold in fold_results1]
    val_auc_scores_nomlm = [fold['best_val_auc'] for fold in fold_results2]

    train_p_value = ttest_rel(train_auc_scores_mlm, train_auc_scores_nomlm).pvalue
    val_p_value = ttest_rel(val_auc_scores_mlm, val_auc_scores_nomlm).pvalue

    df_train = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(len(val_auc_scores_mlm))],
        'with MLM': train_auc_scores_mlm,
        'without MLM': train_auc_scores_nomlm,
    })

    df_valid = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(len(val_auc_scores_mlm))],
        'with MLM': val_auc_scores_mlm,
        'without MLM': val_auc_scores_nomlm
    })
    f = plt.figure(figsize=(12, 8))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_train, palette=["#1f77b4", "#ff7f0e"])  # Custom colors
    plt.title(f'Train AUC Comparison (p-value = {train_p_value:.4f})')
    plt.ylabel('AUC')
    plt.ylim(0.5, 1)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_valid, palette=["#2ca02c", "#d62728"])  # Custom colors
    plt.title(f'Validation AUC Comparison (p-value = {val_p_value:.4f})')
    plt.ylabel('AUC')
    plt.ylim(0.5, 1)

    f.savefig('./figures/auc_comparison.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_loss_comparison_mlm_vs_nomlm(fold_results1, fold_results2, title="Loss Comparison"):
    """Plot loss comparison between two models."""

    f = plt.figure(figsize=(12, 8))

    for i, fold in enumerate(fold_results1):
        train_losses_mlm = fold['metrics']['train_loss']
        val_losses_mlm = fold['metrics']['val_loss']
        train_losses_nomlm = fold_results2[i]['metrics']['train_loss']
        val_losses_nomlm = fold_results2[i]['metrics']['val_loss']
        epochs = range(1, len(train_losses_mlm) + 1)
        
        plt.plot(epochs, train_losses_mlm, 'o-', label=f'Train Loss w/ Pre-Training - Fold {fold["fold"]}', alpha=0.5)
        plt.plot(epochs, val_losses_mlm, 'x-', label=f'Validation Loss w/ Pre-Training - Fold {fold["fold"]}', alpha=0.5)
        plt.plot(epochs, train_losses_nomlm, 'o--', label=f'Train Loss w/o Pre-Training - Fold {fold["fold"]}', alpha=0.5)
        plt.plot(epochs, val_losses_nomlm, 'x--', label=f'Validation Loss w/o Pre-Training - Fold {fold["fold"]}', alpha=0.5)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        f.savefig('./figures/loss_comparison.pdf', bbox_inches='tight')
        plt.show()

def plot_fold_losses(fold_results, title="Losses"):
    """Plot loss for each fold."""

    f = plt.figure(figsize=(12, 8))

    for i, fold in enumerate(fold_results):
        train_losses = fold['metrics']['train_loss']
        val_losses = fold['metrics']['val_loss']
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'o-', label=f'Train Loss - Fold {fold["fold"]}', alpha=0.5)
        plt.plot(epochs, val_losses, 'x-', label=f'Validation Loss - Fold {fold["fold"]}', alpha=0.5)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    f.savefig('./figures/fold_losses.pdf', bbox_inches='tight')
    plt.show()

def plot_data_distribution(adata_RNA, adata_ATAC, adata_Flux, title="Data Distribution"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    plt.suptitle(title)
    
    data = adata_RNA.X.toarray().flatten()
    sns.histplot(data, bins=100, ax=axes[0], color='skyblue')
    var, mean = data.var(), data.mean()
    axes[0].set_title(f'RNA Distribution, var:{var:.2f}, mean:{mean:.2f}')
    axes[0].set_xlabel('Expression level')
    axes[0].set_ylabel('Frequency')

    data = adata_ATAC.X.toarray().flatten()
    sns.histplot(data, bins=100, ax=axes[1], color='lightgreen')
    var, mean = data.var(), data.mean()
    axes[1].set_title(f'ATAC Distribution, var:{var:.3f}, mean:{mean:.2f}')
    axes[1].set_xlabel('Accessibility level')
    axes[1].set_ylabel('Frequency')

    data = adata_Flux.values.flatten()
    sns.histplot(data, bins=100, ax=axes[2], color='salmon')
    var, mean = data.var(), data.mean()
    axes[2].set_title(f'Fluxomic Distribution, var:{var:.5f}, mean:{mean:.2f}')
    axes[2].set_xlabel('Flux value')
    axes[2].set_ylabel('Frequency')

    fig.savefig('./figures/data_distribution.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plot_att_weights(all_attention, dead_end_attention, reprogramming_attention, 
                    feature_names=None, print_top_features=False, top_n=5, scale_weights=False, fix_scale=False,
                    use_mean_contribution=False):


    print(all_attention.shape, "all_attention.shape")
    print(dead_end_attention.shape, "dead_end_attention.shape")
    print(reprogramming_attention.shape, "reprogramming_attention.shape")
    def minmax_scale(arr):
        arr = np.asarray(arr)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr)  # avoid divide by zero
        return (arr - min_val) / (max_val - min_val)

    avg_all_attention = all_attention.mean(axis=0) # Average attention weights across samples
    avg_dead_end_attention = dead_end_attention.mean(axis=0)
    avg_reprogramming_attention = reprogramming_attention.mean(axis=0)

    # Store original unscaled versions for modality contribution calculation
    avg_all_attention_orig = avg_all_attention.copy() if hasattr(avg_all_attention, 'copy') else np.array(avg_all_attention)
    avg_dead_end_attention_orig = avg_dead_end_attention.copy() if hasattr(avg_dead_end_attention, 'copy') else np.array(avg_dead_end_attention)
    avg_reprogramming_attention_orig = avg_reprogramming_attention.copy() if hasattr(avg_reprogramming_attention, 'copy') else np.array(avg_reprogramming_attention)

    if scale_weights:
        avg_all_attention = minmax_scale(avg_all_attention)
        avg_dead_end_attention = minmax_scale(avg_dead_end_attention)
        avg_reprogramming_attention = minmax_scale(avg_reprogramming_attention)
        vmin, vmax = 0.0, 1.0
    elif fix_scale: #  fix scale of all attention weights to the same range
        vmin, vmax = avg_all_attention.min(), avg_all_attention.max()
    else:
        vmin, vmax = None, None

    # Visualize average attention weights
    f = plt.figure(figsize=(15, 3))

    divider1 = 945
    divider2 = 945 + 884

    def add_modality_labels(ax, attention_weights, attention_weights_orig, use_mean=False):
        
        rna_weights = attention_weights_orig[:divider1]
        atac_weights = attention_weights_orig[divider1:divider2]
        flux_weights = attention_weights_orig[divider2:]
        
        # Calculate metric based on method
        if use_mean is False or use_mean == 'sum':
            # Sum of all attention weights (original behavior)
            rna_metric = rna_weights.sum()
            atac_metric = atac_weights.sum()
            flux_metric = flux_weights.sum()
            
        elif use_mean is True or use_mean == 'mean':
            # Mean attention per feature
            rna_metric = rna_weights.mean()
            atac_metric = atac_weights.mean()
            flux_metric = flux_weights.mean()
            
        elif use_mean == 'median':
            # Median attention per feature (robust to zeros and outliers)
            rna_metric = np.median(rna_weights)
            atac_metric = np.median(atac_weights)
            flux_metric = np.median(flux_weights)
            
        elif use_mean == 'trimmed_mean':
            # Trimmed mean: exclude lowest 25% and highest 5%
            rna_metric = stats.trim_mean(rna_weights, proportiontocut=0.15)  # removes 15% from each tail
            atac_metric = stats.trim_mean(atac_weights, proportiontocut=0.15)
            flux_metric = stats.trim_mean(flux_weights, proportiontocut=0.15)
            
        elif use_mean == 'active_mean':
            # Mean of only "active" features (attention > threshold)
            threshold = np.percentile(attention_weights_orig, 25)  # bottom 25% considered inactive
            
            rna_active = rna_weights[rna_weights > threshold]
            atac_active = atac_weights[atac_weights > threshold]
            flux_active = flux_weights[flux_weights > threshold]
            
            rna_metric = rna_active.mean() if len(rna_active) > 0 else 0
            atac_metric = atac_active.mean() if len(atac_active) > 0 else 0
            flux_metric = flux_active.mean() if len(flux_active) > 0 else 0
            
        else:
            raise ValueError(f"Invalid use_mean value: {use_mean}")
        
        # # Normalize to percentages
        # print(rna_metric, atac_metric, flux_metric, "rna_metric, atac_metric, flux_metric")
        # total_metric = rna_metric + atac_metric + flux_metric
        # rna_pct = (rna_metric / total_metric * 100) if total_metric > 0 else 0
        # atac_pct = (atac_metric / total_metric * 100) if total_metric > 0 else 0
        # flux_pct = (flux_metric / total_metric * 100) if total_metric > 0 else 0
        
        # Calculate center positions for each modality
        n_rna = divider1
        n_atac = divider2 - divider1
        n_flux = len(attention_weights) - divider2
        rna_center = n_rna / 2
        atac_center = divider1 + n_atac / 2
        flux_center = divider2 + n_flux / 2
        rna_metric_mean = rna_metric / n_rna
        atac_metric_mean = atac_metric / n_atac
        flux_metric_mean = flux_metric / n_flux
        
        ax.text(rna_center, -0.3, f'Sum: {rna_metric:.3f}\nMean: {rna_metric_mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(atac_center, -0.3, f'Sum: {atac_metric:.3f}\nMean: {atac_metric_mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(flux_center, -0.3, f'Sum: {flux_metric:.3f}\nMean: {flux_metric_mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.subplot(1, 3, 1)
    ax1 = plt.gca()
    sns.heatmap(avg_all_attention.reshape(1, -1), cmap='viridis',  yticklabels=['All'], vmin=vmin, vmax=vmax, ax=ax1)
    plt.title('Avg Att. W. (All Samples)')
    plt.xlabel('Features')
    plt.xticks([])
    plt.axvline(x=divider1, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=divider2, color='red', linestyle='--', linewidth=2)
    add_modality_labels(ax1, avg_all_attention, avg_all_attention_orig, use_mean=use_mean_contribution)

    plt.subplot(1, 3, 2)
    ax2 = plt.gca()
    sns.heatmap(avg_dead_end_attention.reshape(1, -1), cmap='viridis', yticklabels=['Dead-end'], vmin=vmin, vmax=vmax, ax=ax2)
    plt.title('Avg Att. W.  (Dead-end Samples)')
    plt.xlabel('Features')
    plt.xticks([])
    plt.axvline(x=divider1, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=divider2, color='red', linestyle='--', linewidth=2)
    add_modality_labels(ax2, avg_dead_end_attention, avg_dead_end_attention_orig, use_mean=use_mean_contribution)

    plt.subplot(1, 3, 3)
    ax3 = plt.gca()
    sns.heatmap(avg_reprogramming_attention.reshape(1, -1), cmap='viridis', yticklabels=['Reprogramming'], vmin=vmin, vmax=vmax, ax=ax3)
    plt.title('Avg Att. W. (Reprogramming Samples)')
    plt.xlabel('Features')
    plt.xticks([])
    plt.axvline(x=divider1, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=divider2, color='red', linestyle='--', linewidth=2)
    add_modality_labels(ax3, avg_reprogramming_attention, avg_reprogramming_attention_orig, use_mean=use_mean_contribution)

    # f.savefig('./figures/attention_weights.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    if print_top_features:
        def get_top_features(attention_weights, feature_names, top_n=top_n):
            avg_attention = attention_weights.mean(axis=0).numpy() if hasattr(attention_weights, 'numpy') else attention_weights.mean(axis=0)
            print(avg_attention.shape, len(feature_names))
            top_indices = avg_attention.argsort()[-top_n:][::-1]
            print(top_indices)
            return [(feature_names[i], avg_attention[i]) for i in top_indices]

        top_all = get_top_features(all_attention, feature_names)
        top_dead_end = get_top_features(dead_end_attention, feature_names)
        top_reprogramming = get_top_features(reprogramming_attention, feature_names)

        print(f"Top {top_n} attended features (All samples):")
        for feature, weight in top_all:
            print(f"{feature}: {weight:.4f}", end=", ")

        print(f"\nTop {top_n} attended features (Dead-end samples):")
        for feature, weight in top_dead_end:
            print(f"{feature}: {weight:.4f}", end=", ")

        print(f"\nTop {top_n} attended features (Reprogramming samples):")
        for feature, weight in top_reprogramming:
            print(f"{feature}: {weight:.4f}", end=", ")
    return f

def plot_att_weights_distribution(
    all_attention, dead_end_attention, reprogramming_attention,
    feature_names=None, plot_type='violin', top_n=5, print_means=False
):
    divider1 = 944  # RNA ends
    divider2 = 944 + 883  # ATAC ends, Flux begins
    divider1 = 945
    divider2 = 945 + 884

    # Prepare data for plotting
    def prepare_modality_data(attention_weights, condition_name):
        """Extract attention weights by modality"""
        rna_weights = attention_weights[:, :divider1].flatten()
        atac_weights = attention_weights[:, divider1:divider2].flatten()
        flux_weights = attention_weights[:, divider2:].flatten()
        return {
            'RNA': rna_weights,
            'ATAC': atac_weights,
            'Flux': flux_weights,
            'condition': condition_name,
        }

    all_data = prepare_modality_data(all_attention, 'All')
    de_data = prepare_modality_data(dead_end_attention, 'Dead-end')
    re_data = prepare_modality_data(reprogramming_attention, 'Reprogramming')

    if plot_type in ['violin', 'box']:
        # Create DataFrame for seaborn plotting
        data_list = []
        for condition_data in [all_data, de_data, re_data]:
            condition = condition_data['condition']
            for modality in ['RNA', 'ATAC', 'Flux']:
                weights = condition_data[modality]
                for weight in weights:
                    data_list.append({
                        'Condition': condition,
                        'Modality': modality,
                        'Attention Weight': weight
                    })
        df = pd.DataFrame(data_list)
        
        # Create figure with subplots for each condition
        f, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        conditions = ['All', 'Dead-end', 'Reprogramming']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # RNA, ATAC, Flux colors

        # Optionally print means
        if print_means:
            print("Mean attention weight values per modality and per condition:")
        
        for idx, (ax, condition) in enumerate(zip(axes, conditions)):
            condition_df = df[df['Condition'] == condition]
            
            if plot_type == 'violin':
                sns.violinplot(data=condition_df, x='Modality', y='Attention Weight', 
                             palette=colors, ax=ax)
            else:  # box
                sns.boxplot(data=condition_df, x='Modality', y='Attention Weight', 
                          palette=colors, ax=ax)
            
            ax.set_title(f'{condition} Samples', fontsize=12, fontweight='bold')
            ax.set_xlabel('Modality', fontsize=11)
            ax.set_ylabel('Attention Weight', fontsize=11)
            ax.grid(axis='y', alpha=0.3)

            for i, modality in enumerate(['RNA', 'ATAC', 'Flux']):
                mod_data = condition_df[condition_df['Modality'] == modality]['Attention Weight']
                mean_val = mod_data.mean()
                std_val = mod_data.std()
                ax.hlines(mean_val, i - 0.4, i + 0.4, colors='red', linestyles='--', 
                          linewidth=2, alpha=0.7, label='Mean' if i == 0 else '')
                if print_means:
                    print(f"{condition} - {modality}: mean={mean_val:.8f}, std={std_val:.8f}")
            
            if idx == 0:
                ax.legend()
        
    else:
        raise ValueError(f"plot_type must be 'violin', 'box', or 'hist', got '{plot_type}'")
    
    plt.tight_layout()
    plt.show()
    
    return f

def plot_att_heads(all_attention_heads, dead_end_attention_heads, reprogramming_attention_heads, stacked=False):
    n_heads = all_attention_heads.shape[1]  # Assuming the second dimension is the number of heads
    
    if stacked:

        # Visualize stacked attention weights
        f = plt.figure(figsize=(15, 10))  # Adjust figure size

        # Plot for "All Samples" attention weights (stacked)
        plt.subplot(1, 3, 1)
        stacked_all_attention = all_attention_heads.mean(axis=0).reshape(n_heads, -1)  # Stack attention heads
        sns.heatmap(stacked_all_attention, cmap='viridis', yticklabels=[f'Head {i+1}' for i in range(n_heads)])
        plt.title('Stacked Attention Weights (All Samples)')
        plt.xlabel('Features')
        plt.ylabel('Heads')
        plt.xticks(rotation=90)

        # Plot for "Dead-end Samples" attention weights (stacked)
        plt.subplot(1, 3, 2)
        stacked_dead_end_attention = dead_end_attention_heads.mean(axis=0).reshape(n_heads, -1)
        sns.heatmap(stacked_dead_end_attention, cmap='viridis', yticklabels=[f'Head {i+1}' for i in range(n_heads)])
        plt.title('Stacked Attention Weights (Dead-end Samples)')
        plt.xlabel('Features')
        plt.ylabel('Heads')
        plt.xticks(rotation=90)

        # Plot for "Reprogramming Samples" attention weights (stacked)
        plt.subplot(1, 3, 3)
        stacked_reprogramming_attention = reprogramming_attention_heads.mean(axis=0).reshape(n_heads, -1)
        sns.heatmap(stacked_reprogramming_attention, cmap='viridis', yticklabels=[f'Head {i+1}' for i in range(n_heads)])
        plt.title('Stacked Attention Weights (Reprogramming Samples)')
        plt.xlabel('Features')
        plt.ylabel('Heads')
        plt.xticks(rotation=90)

        f.savefig('./figures/attention_heads_stacked.pdf', bbox_inches='tight')
        plt.tight_layout()
        plt.show() 
    
    else:
        # Visualize attention weights for each head
        f = plt.figure(figsize=(15, 15))  # Adjusting the figure size to accommodate more subplots

        # Plot for "All Samples" attention weights
        for head in range(n_heads):
            plt.subplot(n_heads, 3, 3 * head + 1)  # (n_heads rows, 3 columns for each category)
            sns.heatmap(all_attention_heads[:, head, :].mean(axis=0).reshape(1, -1), cmap='viridis', yticklabels=[f'Head {head+1}'])
            plt.title(f'Head {head+1} Attention (All Samples)')
            plt.xlabel('Features')
            plt.xticks(rotation=90)

            # Plot for "Dead-end Samples" attention weights
            plt.subplot(n_heads, 3, 3 * head + 2)
            sns.heatmap(dead_end_attention_heads[:, head, :].mean(axis=0).reshape(1, -1), cmap='viridis', yticklabels=[f'Head {head+1}'])
            plt.title(f'Head {head+1} Attention (Dead-end Samples)')
            plt.xlabel('Features')
            plt.xticks(rotation=90)

            # Plot for "Reprogramming Samples" attention weights
            plt.subplot(n_heads, 3, 3 * head + 3)
            sns.heatmap(reprogramming_attention_heads[:, head, :].mean(axis=0).reshape(1, -1), cmap='viridis', yticklabels=[f'Head {head+1}'])
            plt.title(f'Head {head+1} Attention (Reprogramming Samples)')
            plt.xlabel('Features')
            plt.xticks(rotation=90)

        f.savefig('./figures/attention_heads.pdf', bbox_inches='tight')
        plt.tight_layout()
        plt.show()

