import pandas as pd
import numpy as np
import torch
import os
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def optimize_weights(args):
    """Optimize weights between embedding similarity and disease importance"""
    print("Optimizing weights between embedding similarity and disease importance...")
    
    # Load embeddings and mappings
    embeddings_file = 'results/embeddings/node_embeddings.pt'
    node_map_file = 'data/processed/node_map.json'
    reverse_node_map_file = 'data/processed/reverse_node_map.json'
    
    if not all(os.path.exists(f) for f in [embeddings_file, node_map_file, reverse_node_map_file]):
        print("Required files not found. Please run training first.")
        return
    
    embeddings = torch.load(embeddings_file)
    
    with open(node_map_file, 'r') as f:
        node_map = {k: int(v) for k, v in json.load(f).items()}
    
    with open(reverse_node_map_file, 'r') as f:
        reverse_node_map = {int(k): v for k, v in json.load(f).items()}
    
    # Load validated reference data
    reference_file = 'data/processed/validated_reference.csv'
    if not os.path.exists(reference_file):
        print("Validated reference data not found.")
        return
    
    reference_df = pd.read_csv(reference_file)
    
    # Load disease importance data
    disease_file = 'data/raw/disease_importance_extended.csv'
    if not os.path.exists(disease_file):
        print("Disease importance data not found.")
        return
    
    disease_df = pd.read_csv(disease_file)
    
    # Create disease importance dictionary
    disease_importance = {}
    target_importance = {}
    
    for _, row in disease_df.iterrows():
        target = row['target']
        importance = row['importance_score']
        disease = row['disease']
        
        if target in node_map:
            target_idx = node_map[target]
            
            if target_idx not in target_importance:
                target_importance[target_idx] = []
            
            target_importance[target_idx].append((disease, importance))
    
    # Calculate average importance for each target
    for target_idx, importances in target_importance.items():
        scores = [score for _, score in importances]
        disease_importance[target_idx] = np.mean(scores) if scores else 0.0
    
    # Get targets for evaluation
    target_nodes = [name for name, idx in node_map.items() 
                   if isinstance(name, str) and any(name.startswith(prefix) 
                                                  for prefix in ['Target_', 'IL', 'TNF', 'CASP', 'MMP', 'BCL', 'P'])]
    
    # Try different embedding_weight values
    weights = np.linspace(0.0, 1.0, args.steps)
    results = []
    
    for embedding_weight in tqdm(weights):
        importance_weight = 1.0 - embedding_weight
        
        # Calculate metrics for this weight combination
        all_ranks = []
        all_reciprocal_ranks = []
        hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
        
        for _, row in reference_df.iterrows():
            compound = row['compound']
            target = row['target']
            
            if compound not in node_map or target not in node_map:
                continue
            
            compound_idx = node_map[compound]
            target_idx = node_map[target]
            
            # Get compound embedding
            compound_emb = embeddings[compound_idx].unsqueeze(0)
            
            # Calculate similarity with all targets
            similarities = []
            for target_name in target_nodes:
                if target_name in node_map:
                    curr_target_idx = node_map[target_name]
                    curr_target_emb = embeddings[curr_target_idx].unsqueeze(0)
                    
                    # Calculate cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(
                        compound_emb, curr_target_emb, dim=1).item()
                    
                    # Get target importance
                    importance_score = disease_importance.get(curr_target_idx, 0.0)
                    
                    # Calculate combined score
                    combined_score = (
                        embedding_weight * similarity +
                        importance_weight * importance_score
                    )
                    
                    similarities.append((target_name, combined_score))
            
            # Sort by combined score
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            
            # Find rank of the true target
            rank = -1
            for i, (t_name, _) in enumerate(similarities):
                if t_name == target:
                    rank = i + 1
                    break
            
            if rank == -1:
                continue
            
            # Record metrics
            all_ranks.append(rank)
            all_reciprocal_ranks.append(1.0 / rank)
            
            for k in hits_at_k.keys():
                if rank <= k:
                    hits_at_k[k] += 1
        
        # Calculate average metrics
        if all_ranks:
            mrr = np.mean(all_reciprocal_ranks)
            mean_rank = np.mean(all_ranks)
            mean_normalized_rank = mean_rank / len(target_nodes)
            
            total_pairs = len(all_ranks)
            hit_rates = {k: count/total_pairs for k, count in hits_at_k.items()}
            
            results.append({
                'embedding_weight': embedding_weight,
                'importance_weight': importance_weight,
                'MRR': mrr,
                'Mean_Rank': mean_rank,
                'Mean_Normalized_Rank': mean_normalized_rank,
                'Hit@1': hit_rates[1],
                'Hit@3': hit_rates[3],
                'Hit@5': hit_rates[5],
                'Hit@10': hit_rates[10],
                'Hit@20': hit_rates[20]
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs('results/weight_optimization', exist_ok=True)
    results_df.to_csv('results/weight_optimization/weight_results.csv', index=False)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_df['embedding_weight'], results_df['MRR'], marker='o')
    plt.title('MRR vs Embedding Weight')
    plt.xlabel('Embedding Weight')
    plt.ylabel('MRR')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_df['embedding_weight'], results_df['Mean_Normalized_Rank'], marker='o')
    plt.title('Mean Normalized Rank vs Embedding Weight')
    plt.xlabel('Embedding Weight')
    plt.ylabel('Mean Normalized Rank')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    for k in [1, 3, 5, 10, 20]:
        plt.plot(results_df['embedding_weight'], results_df[f'Hit@{k}'], 
                marker='o', label=f'Hit@{k}')
    
    plt.title('Hit@k vs Embedding Weight')
    plt.xlabel('Embedding Weight')
    plt.ylabel('Hit Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/weight_optimization/weight_optimization.png', dpi=300)
    
    # Find optimal weight
    if args.optimize_for == 'mrr':
        best_idx = results_df['MRR'].idxmax()
    elif args.optimize_for == 'hit1':
        best_idx = results_df['Hit@1'].idxmax()
    elif args.optimize_for == 'hit5':
        best_idx = results_df['Hit@5'].idxmax()
    elif args.optimize_for == 'hit10':
        best_idx = results_df['Hit@10'].idxmax()
    else:
        best_idx = results_df['MRR'].idxmax()
    
    best_weights = results_df.iloc[best_idx]
    
    print("\n===== Optimal Weight Configuration =====")
    print(f"Optimized for: {args.optimize_for}")
    print(f"Embedding Weight: {best_weights['embedding_weight']:.4f}")
    print(f"Importance Weight: {best_weights['importance_weight']:.4f}")
    print("\nPerformance with Optimal Weights:")
    print(f"MRR: {best_weights['MRR']:.4f}")
    print(f"Mean Rank: {best_weights['Mean_Rank']:.2f}")
    print(f"Mean Normalized Rank: {best_weights['Mean_Normalized_Rank']:.4f}")
    print(f"Hit@1: {best_weights['Hit@1']:.4f}")
    print(f"Hit@5: {best_weights['Hit@5']:.4f}")
    print(f"Hit@10: {best_weights['Hit@10']:.4f}")
    print(f"Hit@20: {best_weights['Hit@20']:.4f}")
    
    # Save optimal weights
    with open('results/weight_optimization/optimal_weights.txt', 'w') as f:
        f.write(f"Optimized for: {args.optimize_for}\n")
        f.write(f"Embedding Weight: {best_weights['embedding_weight']:.4f}\n")
        f.write(f"Importance Weight: {best_weights['importance_weight']:.4f}\n")
        f.write("\nPerformance with Optimal Weights:\n")
        f.write(f"MRR: {best_weights['MRR']:.4f}\n")
        f.write(f"Mean Rank: {best_weights['Mean_Rank']:.2f}\n")
        f.write(f"Mean Normalized Rank: {best_weights['Mean_Normalized_Rank']:.4f}\n")
        f.write(f"Hit@1: {best_weights['Hit@1']:.4f}\n")
        f.write(f"Hit@5: {best_weights['Hit@5']:.4f}\n")
        f.write(f"Hit@10: {best_weights['Hit@10']:.4f}\n")
        f.write(f"Hit@20: {best_weights['Hit@20']:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize weights between embedding similarity and disease importance")
    
    parser.add_argument('--steps', type=int, default=11,
                      help='Number of weight steps to evaluate (default: 11)')
    parser.add_argument('--optimize_for', type=str, default='mrr', choices=['mrr', 'hit1', 'hit5', 'hit10'],
                      help='Metric to optimize for (default: mrr)')
    
    args = parser.parse_args()
    optimize_weights(args)
