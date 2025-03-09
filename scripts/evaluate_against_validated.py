import pandas as pd
import numpy as np
import torch
import argparse
import os
import json
from tqdm import tqdm
from src.evaluation.metrics import calculate_hit_at_k, calculate_mrr

def load_embeddings_and_mappings():
    """Load embeddings and node mappings"""
    # Check if files exist
    if not os.path.exists('results/embeddings/node_embeddings.pt'):
        print("Embeddings file not found. Please run training first.")
        return None, None, None
    
    if not os.path.exists('data/processed/node_map.json') or \
       not os.path.exists('data/processed/reverse_node_map.json'):
        print("Node mapping files not found. Please run training first.")
        return None, None, None
    
    # Load embeddings
    embeddings = torch.load('results/embeddings/node_embeddings.pt')
    
    # Load node mappings
    with open('data/processed/node_map.json', 'r') as f:
        node_map = json.load(f)
        # Convert string keys to original types
        node_map = {k: int(v) for k, v in node_map.items()}
    
    with open('data/processed/reverse_node_map.json', 'r') as f:
        reverse_node_map = json.load(f)
        # Ensure keys are integers
        reverse_node_map = {int(k): v for k, v in reverse_node_map.items()}
    
    return embeddings, node_map, reverse_node_map

def load_validated_reference():
    """Load validated reference data"""
    reference_file = 'data/processed/validated_reference.csv'
    if not os.path.exists(reference_file):
        print("Validated reference data not found. Please run integrate_validated_data.py first.")
        return None
    
    return pd.read_csv(reference_file)

def evaluate_against_validated(embeddings, node_map, reference_df, disease=None, embedding_weight=0.6, importance_weight=0.4):
    """Evaluate model against validated targets"""
    print("Evaluating model against validated targets...")
    
    # Load disease importance data if available
    disease_importance = {}
    if os.path.exists('data/raw/disease_importance_extended.csv'):
        disease_df = pd.read_csv('data/raw/disease_importance_extended.csv')
        
        # Filter by disease if specified
        if disease:
            disease_df = disease_df[disease_df['disease'] == disease]
        
        # Create target importance mapping
        for _, row in disease_df.iterrows():
            target = row['target']
            importance = row['importance_score']
            
            if target in node_map:
                target_idx = node_map[target]
                disease_importance[target_idx] = importance
    
    # Initialize metrics
    all_ranks = []
    all_reciprocal_ranks = []
    hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    compound_target_pairs = []
    
    # Get all target nodes
    target_nodes = [name for name, idx in node_map.items() 
                   if isinstance(name, str) and any(name.startswith(prefix) 
                                                   for prefix in ['Target_', 'IL', 'TNF', 'CASP', 'MMP', 'BCL', 'P', 'S'])]
    
    # Evaluate each validated compound-target pair
    for _, row in tqdm(reference_df.iterrows(), total=len(reference_df)):
        compound = row['compound']
        target = row['target']
        
        # Check if compound and target are in node map
        if compound not in node_map or target not in node_map:
            print(f"Compound {compound} or target {target} not in node map, skipping...")
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
        
        # Find true target's rank
        rank = -1
        for i, (t_name, _) in enumerate(similarities):
            if t_name == target:
                rank = i + 1  # Rank starts at 1
                break
        
        if rank == -1:
            print(f"Warning: Target {target} not found in ranking list, which should not happen.")
            continue
        
        # Record metrics
        all_ranks.append(rank)
        all_reciprocal_ranks.append(1.0 / rank)
        
        # Calculate Hit@k
        for k in hits_at_k.keys():
            if rank <= k:
                hits_at_k[k] += 1
        
        compound_target_pairs.append((compound, target, rank))
    
    # Calculate average metrics
    if len(all_ranks) > 0:
        # Main metrics
        mrr = np.mean(all_reciprocal_ranks)
        mean_rank = np.mean(all_ranks)
        mean_normalized_rank = mean_rank / len(target_nodes)
        
        # Hit@k rates
        total_pairs = len(all_ranks)
        hit_rates = {k: count/total_pairs for k, count in hits_at_k.items()}
        
        # Print results
        print("\n===== Validated Target Evaluation Results =====")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f"Mean Rank: {mean_rank:.2f}")
        print(f"Mean Normalized Rank: {mean_normalized_rank:.4f}")
        for k, rate in hit_rates.items():
            print(f"Hit@{k}: {rate:.4f}")
        
        # Save to CSV
        results_df = pd.DataFrame({
            'MRR': [mrr],
            'Mean Rank': [mean_rank],
            'Mean Normalized Rank': [mean_normalized_rank],
            'Hit@1': [hit_rates[1]],
            'Hit@3': [hit_rates[3]],
            'Hit@5': [hit_rates[5]],
            'Hit@10': [hit_rates[10]],
            'Hit@20': [hit_rates[20]]
        })
        
        # Save detailed rankings
        pairs_df = pd.DataFrame(compound_target_pairs, 
                               columns=['Compound', 'Target', 'Rank'])
        
        # Create results directory
        os.makedirs('results/evaluation', exist_ok=True)
        
        # Save results
        results_df.to_csv('results/evaluation/validated_metrics.csv', index=False)
        pairs_df.to_csv('results/evaluation/validated_rankings.csv', index=False)
        
        return results_df
    else:
        print("No compound-target pairs were successfully evaluated.")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate against validated targets")
    parser.add_argument('--disease', type=str, default=None,
                        help='Specific disease to evaluate for')
    parser.add_argument('--embedding_weight', type=float, default=0.6,
                        help='Weight for embedding similarity')
    parser.add_argument('--importance_weight', type=float, default=0.4,
                        help='Weight for disease importance')
    args = parser.parse_args()
    
    print("Starting evaluation against validated targets...")
    
    # Load embeddings and mappings
    embeddings, node_map, reverse_node_map = load_embeddings_and_mappings()
    if embeddings is None:
        return
    
    # Load validated data
    reference_df = load_validated_reference()
    if reference_df is None:
        return
    
    # Evaluate
    results = evaluate_against_validated(
        embeddings, 
        node_map, 
        reference_df, 
        args.disease,
        args.embedding_weight,
        args.importance_weight
    )
    
    if results is not None:
        print("\nEvaluation complete! Results saved to results/evaluation/ directory.")
    else:
        print("\nEvaluation failed. Please check data and logs.")

if __name__ == "__main__":
    main()
