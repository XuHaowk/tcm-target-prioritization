import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate_model(prioritized_targets, validation_set, top_k=10):
    """
    Evaluate target prioritization model performance.
    
    Args:
        prioritized_targets: List of (target_idx, score) tuples, sorted by score
        validation_set: List of correct target indices
        top_k: K value for precision@k
    
    Returns:
        Dictionary of performance metrics
    """
    # Extract target indices and scores
    target_indices = [t[0] for t in prioritized_targets]
    target_scores = [t[1] for t in prioritized_targets]
    
    # Create binary labels (1 for correct targets, 0 otherwise)
    y_true = np.zeros(len(target_indices))
    for i, idx in enumerate(target_indices):
        if idx in validation_set:
            y_true[i] = 1
    
    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, target_scores)
    except:
        roc_auc = 0  # Handle edge cases
    
    # Calculate PR AUC
    precision, recall, _ = precision_recall_curve(y_true, target_scores)
    pr_auc = auc(recall, precision)
    
    # Calculate precision@k
    top_k_indices = target_indices[:top_k]
    top_k_hits = sum(1 for idx in top_k_indices if idx in validation_set)
    precision_at_k = top_k_hits / top_k
    
    # Calculate average rank of valid targets
    valid_ranks = []
    for val_idx in validation_set:
        try:
            rank = target_indices.index(val_idx) + 1
            valid_ranks.append(rank)
        except ValueError:
            # If a validation target is not in our ranking list, assign max rank
            valid_ranks.append(len(target_indices))
    
    mean_rank = np.mean(valid_ranks) if valid_ranks else 0
    
    # Return metrics dictionary
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        f'precision@{top_k}': precision_at_k,
        'mean_rank': mean_rank
    }
    
    return metrics

def calculate_hit_at_k(prioritized_targets, validation_set, k_values=[1, 3, 5, 10, 20]):
    """
    Calculate Hit@k metrics for different k values.
    
    Args:
        prioritized_targets: List of (target_idx, score) tuples, sorted by score
        validation_set: List of correct target indices
        k_values: List of k values to calculate Hit@k for
    
    Returns:
        Dictionary mapping k to Hit@k values
    """
    # Extract target indices
    target_indices = [t[0] for t in prioritized_targets]
    
    results = {}
    for k in k_values:
        # Calculate Hit@k
        top_k_indices = target_indices[:k]
        hits = sum(1 for idx in top_k_indices if idx in validation_set)
        hit_at_k = hits / len(validation_set) if validation_set else 0
        
        results[f'Hit@{k}'] = hit_at_k
    
    return results

def calculate_mrr(prioritized_targets, validation_set):
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        prioritized_targets: List of (target_idx, score) tuples, sorted by score
        validation_set: List of correct target indices
    
    Returns:
        MRR value
    """
    # Extract target indices
    target_indices = [t[0] for t in prioritized_targets]
    
    reciprocal_ranks = []
    for val_idx in validation_set:
        try:
            rank = target_indices.index(val_idx) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            # If target not found, contribution is 0
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    return mrr
