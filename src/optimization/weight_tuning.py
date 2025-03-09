"""
Weight tuning module for TCM target prioritization

This module provides functions to find optimal weights between 
embedding similarity and target importance scores.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
import torch


def calculate_priorities_with_weights(embedding_scores, importance_scores, embedding_weight):
    """
    Calculate target priority scores with specific weighting
    
    Args:
        embedding_scores: Dictionary mapping compounds to target embedding similarity scores
        importance_scores: Dictionary mapping targets to importance scores
        embedding_weight: Weight for embedding scores (1-embedding_weight will be for importance)
        
    Returns:
        Dictionary mapping compounds to their prioritized targets with new weights
    """
    importance_weight = 1.0 - embedding_weight
    
    # New prioritization dictionary
    new_priorities = {}
    
    # Find median importance score for normalization
    median_importance = 0.5
    if importance_scores:
        importance_values = list(importance_scores.values())
        median_importance = np.median(importance_values)
    
    for compound_id, target_scores in embedding_scores.items():
        compound_priorities = {}
        
        for target_id, embed_score in target_scores.items():
            # Get importance score for this target (default to median if not found)
            importance = importance_scores.get(target_id, median_importance)
            
            # Apply nonlinear transformation to increase contrast
            adjusted_importance = np.power(importance, 1.5)
            
            # Calculate weighted score
            weighted_score = embedding_weight * embed_score + importance_weight * adjusted_importance
            
            # Store in priorities dictionary
            compound_priorities[target_id] = weighted_score
        
        # Sort targets by priority score
        sorted_targets = sorted(compound_priorities.items(), key=lambda x: x[1], reverse=True)
        new_priorities[compound_id] = dict(sorted_targets)
    
    return new_priorities


def grid_search_weights(embedding_scores, importance_scores, validation_matrix, 
                       compound_list, target_list):
    """
    Perform grid search to find optimal weights
    
    Args:
        embedding_scores: Dictionary mapping compounds to target embedding similarity scores
        importance_scores: Dictionary mapping targets to importance scores
        validation_matrix: Binary matrix of validated interactions
        compound_list: List of compound IDs
        target_list: List of target IDs
        
    Returns:
        Dictionary with results for each weight
    """
    # Create maps from IDs to indices
    compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compound_list)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_list)}
    
    # Initialize results
    results = {
        'embedding_weights': [],
        'ap_scores': [],
        'auroc_scores': [],
        'top10_hits': [],
        'top20_hits': [],
        'mrr_scores': []  # Mean Reciprocal Rank
    }
    
    # Create two sets of weight values:
    # 1. Fine-grained values in the 0-0.1 range (since optimal seems to be here)
    # 2. Coarser values for the rest of the range
    fine_weights = np.linspace(0, 0.2, 11)  # 0, 0.02, 0.04, ..., 0.2
    coarse_weights = np.linspace(0.3, 1.0, 8)  # 0.3, 0.4, ..., 1.0
    weight_values = np.concatenate([fine_weights, coarse_weights])
    
    # Try different weights
    for embedding_weight in tqdm(weight_values, desc="Testing weights"):
        # Calculate priorities with this weight
        priorities = calculate_priorities_with_weights(
            embedding_scores, importance_scores, embedding_weight
        )
        
        # Create prediction matrix
        pred_matrix = np.zeros((len(compound_list), len(target_list)))
        
        for comp_id, target_scores in priorities.items():
            if comp_id in compound_to_idx:
                comp_idx = compound_to_idx[comp_id]
                for target_id, score in target_scores.items():
                    if target_id in target_to_idx:
                        target_idx = target_to_idx[target_id]
                        pred_matrix[comp_idx, target_idx] = score
        
        # Calculate metrics
        y_true = validation_matrix.flatten()
        y_score = pred_matrix.flatten()
        
        ap_score = average_precision_score(y_true, y_score)
        auroc_score = roc_auc_score(y_true, y_score)
        
        # Calculate Top-k hits
        top10_hits = 0
        top20_hits = 0
        reciprocal_ranks = []
        total_possible = 0
        
        for comp_idx, comp_id in enumerate(compound_list):
            if comp_id in priorities:
                # Get targets with validation data for this compound
                valid_targets = np.where(validation_matrix[comp_idx] == 1)[0]
                if len(valid_targets) > 0:
                    total_possible += len(valid_targets)
                    
                    # Get predicted targets
                    pred_targets = list(priorities[comp_id].keys())
                    
                    # Calculate reciprocal rank
                    for valid_idx in valid_targets:
                        valid_target = target_list[valid_idx]
                        if valid_target in pred_targets:
                            rank = pred_targets.index(valid_target) + 1
                            reciprocal_ranks.append(1.0 / rank)
                            
                            # Count top-k hits
                            if rank <= 10:
                                top10_hits += 1
                            if rank <= 20:
                                top20_hits += 1
                        else:
                            reciprocal_ranks.append(0.0)
        
        # Calculate MRR and hit rates
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        top10_hit_rate = top10_hits / max(1, total_possible)
        top20_hit_rate = top20_hits / max(1, total_possible)
        
        # Store results
        results['embedding_weights'].append(embedding_weight)
        results['ap_scores'].append(ap_score)
        results['auroc_scores'].append(auroc_score)
        results['top10_hits'].append(top10_hit_rate)
        results['top20_hits'].append(top20_hit_rate)
        results['mrr_scores'].append(mrr)
    
    return results


def plot_weight_tuning_results(results, output_file=None):
    """
    Plot metrics for different weight values
    
    Args:
        results: Dictionary with weight tuning results
        output_file: Path to save the figure (optional)
    """
    # Create a figure with 3 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Precision and ROC subplot
    ax1 = fig.add_subplot(221)
    ax1.plot(results['embedding_weights'], results['ap_scores'], 'b-', label='Average Precision')
    ax1.plot(results['embedding_weights'], results['auroc_scores'], 'r-', label='AUROC')
    
    # Add vertical line at best AP score
    best_ap_idx = np.argmax(results['ap_scores'])
    best_ap_weight = results['embedding_weights'][best_ap_idx]
    best_ap_score = results['ap_scores'][best_ap_idx]
    
    ax1.axvline(x=best_ap_weight, color='gray', linestyle='--')
    ax1.text(best_ap_weight + 0.05, best_ap_score, 
             f'Best AP: {best_ap_score:.3f}\nWeight: {best_ap_weight:.2f}',
             verticalalignment='center')
    
    ax1.set_xlabel('Embedding Weight')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision and ROC Scores vs. Embedding Weight')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Top-10 hit rate subplot
    ax2 = fig.add_subplot(222)
    ax2.plot(results['embedding_weights'], results['top10_hits'], 'g-')
    
    # Add vertical line at best hit rate
    best_hit_idx = np.argmax(results['top10_hits'])
    best_hit_weight = results['embedding_weights'][best_hit_idx]
    best_hit_rate = results['top10_hits'][best_hit_idx]
    
    ax2.axvline(x=best_hit_weight, color='gray', linestyle='--')
    ax2.text(best_hit_weight + 0.05, best_hit_rate, 
             f'Best Hit Rate: {best_hit_rate:.3f}\nWeight: {best_hit_weight:.2f}',
             verticalalignment='center')
    
    ax2.set_xlabel('Embedding Weight')
    ax2.set_ylabel('Top-10 Hit Rate')
    ax2.set_title('Top-10 Hit Rate vs. Embedding Weight')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. MRR subplot
    ax3 = fig.add_subplot(223)
    ax3.plot(results['embedding_weights'], results['mrr_scores'], 'c-')
    
    # Add vertical line at best MRR
    best_mrr_idx = np.argmax(results['mrr_scores'])
    best_mrr_weight = results['embedding_weights'][best_mrr_idx]
    best_mrr_score = results['mrr_scores'][best_mrr_idx]
    
    ax3.axvline(x=best_mrr_weight, color='gray', linestyle='--')
    ax3.text(best_mrr_weight + 0.05, best_mrr_score, 
             f'Best MRR: {best_mrr_score:.3f}\nWeight: {best_mrr_weight:.2f}',
             verticalalignment='center')
    
    ax3.set_xlabel('Embedding Weight')
    ax3.set_ylabel('MRR')
    ax3.set_title('MRR vs Embedding Weight')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Top-20 hit rate subplot
    ax4 = fig.add_subplot(224)
    ax4.plot(results['embedding_weights'], results['top20_hits'], 'm-')
    
    # Add vertical line at best hit rate
    best_hit20_idx = np.argmax(results['top20_hits'])
    best_hit20_weight = results['embedding_weights'][best_hit20_idx]
    best_hit20_rate = results['top20_hits'][best_hit20_idx]
    
    ax4.axvline(x=best_hit20_weight, color='gray', linestyle='--')
    ax4.text(best_hit20_weight + 0.05, best_hit20_rate, 
             f'Best Hit@20: {best_hit20_rate:.3f}\nWeight: {best_hit20_weight:.2f}',
             verticalalignment='center')
    
    ax4.set_xlabel('Embedding Weight')
    ax4.set_ylabel('Top-20 Hit Rate')
    ax4.set_title('Top-20 Hit Rate vs. Embedding Weight')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure first, then display it
    if output_file:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save with high DPI for better quality
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved weight tuning plots to {output_file}")
    
    # If you're running in an interactive environment, show the plot
    # If running in a non-interactive environment, this won't do anything
    plt.close()
    
    # Print summary of optimal weights
    print("\nWeight Tuning Results:")
    print(f"  Optimal weight for Average Precision: {best_ap_weight:.2f} (AP={best_ap_score:.3f})")
    print(f"  Optimal weight for Top-10 hit rate: {best_hit_weight:.2f} (Hit rate={best_hit_rate:.3f})")
    print(f"  Optimal weight for MRR: {best_mrr_weight:.2f} (MRR={best_mrr_score:.3f})")
    print(f"  Optimal weight for Top-20 hit rate: {best_hit20_weight:.2f} (Hit rate={best_hit20_rate:.3f})")
    
    return {
        'optimal_ap_weight': best_ap_weight,
        'optimal_hit_weight': best_hit_weight,
        'optimal_mrr_weight': best_mrr_weight,
        'optimal_hit20_weight': best_hit20_weight
    }


def find_optimal_weights(embedding_scores, importance_scores, validated_data, 
                         compound_list, target_list, output_file=None):
    """
    Find optimal weights and generate a report
    
    Args:
        embedding_scores: Dictionary mapping compounds to target embedding similarity scores
        importance_scores: Dictionary mapping targets to importance scores
        validated_data: DataFrame with validated interactions
        compound_list: List of compound IDs
        target_list: List of target IDs
        output_file: Path to save the visualization (optional)
        
    Returns:
        Dictionary with optimal weights for different metrics
    """
    # Create validation matrix
    validation_matrix = np.zeros((len(compound_list), len(target_list)))
    
    # Create maps from IDs to indices
    compound_to_idx = {comp_id: idx for idx, comp_id in enumerate(compound_list)}
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_list)}
    
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
    
    if interaction_count == 0:
        print("Warning: No validated interactions found for weight tuning")
        return {'optimal_ap_weight': 0.6, 'optimal_hit_weight': 0.6}
    
    # Perform grid search
    results = grid_search_weights(
        embedding_scores, importance_scores, validation_matrix, 
        compound_list, target_list
    )
    
    # Plot results
    optimal_weights = plot_weight_tuning_results(results, output_file)
    
    return optimal_weights
