import torch
from utils.helpers import get_token_embeddings

def compute_similarity_matrix(model, dataset, device):
    """
    Compute the similarity matrix for the dataset.
    Args:
        model (torch.nn.Module): Model.
        dataset (torch.utils.data.Dataset): Dataset.
        device (str): Device to use.
    Returns:
        np.ndarray: Similarity matrix.
    """
    embeddings = get_token_embeddings(model, dataset, device)  # shape: (n_samples, seq_len, d_model)
    
    # Compute the mean embedding for each token across all samples
    mean_token_embeddings = embeddings.mean(dim=0)  # shape: (seq_len, d_model)
    
     # Normalize the mean token embeddings (for cosine similarity)
    mean_token_embeddings = mean_token_embeddings / mean_token_embeddings.norm(dim=1, keepdim=True)
    
    # Calculate cosine similarity for all pairs of tokens using matrix multiplication
    similarity_matrix = torch.mm(mean_token_embeddings, mean_token_embeddings.T).cpu().numpy()
    
    
    return similarity_matrix  # Convert to numpy array if needed