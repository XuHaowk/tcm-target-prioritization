"""
Model validation module for TCM target prioritization

This module provides functions to validate model predictions against known experimental data
and calculate performance metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import torch
from tqdm import tqdm


def load_validation_data(validation_file):
    """
    Load experimentally validated compound-target interactions
    
    Args:
        validation_file: Path to validation data CSV
        
    Returns:
        DataFrame with validated interactions
    """
    try:
        validated_data = pd.read_csv(validation_file)
        print(f"Loaded {len(validated_data)} validated interactions from {validation_file}")
        
        # Standardize column names
        if 'compound' in validated_data.columns and 'compound_id' not in validated_data.columns:
            validated_data['compound_id'] = validated_data['compound']
        
        if 'target' in validated_data.columns and 'target_id' not in validated_data.columns:
            validated_data['target_id'] = validated_data['target']
            
        return validated_data
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return pd.DataFrame()


def create_validation_matrix(validated_data, compounds, targets):
    """
    Create a binary interaction matrix from validation data
    
    Args:
        validated_data: DataFrame with validated interactions
        compounds: List of compound IDs
        targets: List of target IDs
        
    Returns:
        Binary matrix of validated interactions
    """
    # Create a compound-target matrix filled with zeros
    validation_matrix = np.zeros((len(compounds), len(targets)))
    
    # Create maps from IDs to indices
    compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compounds)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(targets)}
    
    # Fill in known interactions
    interaction_count = 0
    
    for _, row in validated_data.iterrows():
        comp_id = str(row.get('compound_id', row.get('compound', '')))
        target_id = str(row.get('target_id', row.get('target', '')))
        
        if comp_id in compound_to_idx and target_id in target_to_idx:
            comp_idx = compound_to_idx[comp_id]
            target_idx = target_to_idx[target_id]
            validation_matrix[comp_idx, target_idx] = 1
            interaction_count += 1
    
    print(f"Created validation matrix with {interaction_count} known interactions")
    
    return validation_matrix


def evaluate_predictions(prediction_scores, validation_matrix):
    """
    Calculate precision, recall, and ROC curves for model predictions
    
    Args:
        prediction_scores: Matrix of predicted scores for compound-target pairs
        validation_matrix: Binary matrix of validated interactions
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Flatten matrices for evaluation
    y_score = prediction_scores.flatten()
    y_true = validation_matrix.flatten()
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Get number of true positives at different thresholds
    true_pos_counts = []
    for threshold in np.linspace(0, 1, 20):
        predicted_pos = (y_score >= threshold).astype(int)
        true_pos = np.sum((predicted_pos == 1) & (y_true == 1))
        true_pos_counts.append(true_pos)
    
    # Calculate hit rates at different k values
    hit_rates = {}
    for k in [1, 3, 5, 10, 20]:
        hit_count = 0
        total_count = 0
        
        for i in range(validation_matrix.shape[0]):
            # Get validated targets for this compound
            valid_targets = np.where(validation_matrix[i] == 1)[0]
            if len(valid_targets) > 0:
                total_count += len(valid_targets)
                
                # Get top k predicted targets
                pred_scores = prediction_scores[i]
                top_indices = np.argsort(pred_scores)[-k:]
                
                # Count hits
                for target_idx in valid_targets:
                    if target_idx in top_indices:
                        hit_count += 1
        
        hit_rates[f'hit@{k}'] = hit_count / max(1, total_count)
    
    return {
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'average_precision': average_precision,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'roc_auc': roc_auc,
        'true_pos_by_threshold': true_pos_counts,
        'hit_rates': hit_rates
    }


def plot_validation_results(metrics, output_file=None):
    """
    Create visualization of validation results
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_file: Path to save the figure (optional)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot precision-recall curve
    ax1.step(metrics['recall'], metrics['precision'], color='b', alpha=0.8, where='post')
    ax1.fill_between(metrics['recall'], metrics['precision'], alpha=0.2, color='b')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(f'Precision-Recall curve (AP={metrics["average_precision"]:.3f})')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot ROC curve
    ax2.plot(metrics['fpr'], metrics['tpr'], color='r', alpha=0.8)
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'ROC curve (AUC={metrics["roc_auc"]:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot hit rates
    hit_rates = metrics['hit_rates']
    k_values = [int(k.split('@')[1]) for k in hit_rates.keys()]
    rate_values = list(hit_rates.values())
    
    ax3.bar(range(len(k_values)), rate_values, tick_label=[f'Hit@{k}' for k in k_values])
    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Rate')
    ax3.set_title('Hit Rates at Different k Values')
    ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add hit rate values above the bars
    for i, v in enumerate(rate_values):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    # Save the figure first, then display it
    if output_file:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save with high DPI for better quality
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved validation plots to {output_file}")
    
    # Close the figure to free memory
    plt.close()
    
    return metrics


def validate_model(predictions, validated_data, compound_list, target_list, output_file=None):
    """
    Validate model predictions and visualize results
    
    Args:
        predictions: Dictionary mapping compounds to their prioritized targets
        validated_data: DataFrame with validated interactions
        compound_list: List of compound IDs
        target_list: List of target IDs
        output_file: Path to save the validation plot (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Create validation matrix
    validation_matrix = create_validation_matrix(validated_data, compound_list, target_list)
    
    # Create prediction score matrix
    prediction_matrix = np.zeros((len(compound_list), len(target_list)))
    
    # Create maps from IDs to indices
    compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compound_list)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_list)}
    
    # Fill in prediction scores
    for comp_id, target_scores in predictions.items():
        if comp_id in compound_to_idx:
            comp_idx = compound_to_idx[comp_id]
            for target_id, score in target_scores.items():
                if target_id in target_to_idx:
                    target_idx = target_to_idx[target_id]
                    prediction_matrix[comp_idx, target_idx] = score
    
    # Evaluate predictions
    metrics = evaluate_predictions(prediction_matrix, validation_matrix)
    
    # Plot results
    if output_file or len(validated_data) > 0:
        metrics = plot_validation_results(metrics, output_file)
    
    return metrics
