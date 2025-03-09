import pandas as pd
import numpy as np
import torch
import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def run_comparative_analysis():
    """Compare model performance with and without disease importance information"""
    print("Running comparative analysis of model performance with and without disease importance...")
    
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
    
    # Function to calculate metrics
    def calculate_metrics(embedding_weight, importance_weight, specific_disease=None):
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
            
            # Skip if target not important for the specific disease
            if specific_disease and target in disease_df['target'].values:
                target_diseases = disease_df[disease_df['target'] == target]['disease'].unique()
                if specific_disease not in target_diseases:
                    continue
            
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
            
            return {
                'embedding_weight': embedding_weight,
                'importance_weight': importance_weight,
                'disease': specific_disease if specific_disease else 'all',
                'total_pairs': total_pairs,
                'MRR': mrr,
                'Mean_Rank': mean_rank,
                'Mean_Normalized_Rank': mean_normalized_rank,
                'Hit@1': hit_rates[1],
                'Hit@3': hit_rates[3],
                'Hit@5': hit_rates[5],
                'Hit@10': hit_rates[10],
                'Hit@20': hit_rates[20]
            }
        else:
            return None
    
    # Run analysis for different configurations
    results = []
    
    # Without disease importance (embedding only)
    print("Evaluating with embedding similarity only...")
    result = calculate_metrics(1.0, 0.0)
    if result:
        results.append(result)
    
    # With optimal balance
    print("Evaluating with optimal weight balance...")
    result = calculate_metrics(0.6, 0.4)
    if result:
        results.append(result)
    
    # With disease importance only
    print("Evaluating with disease importance only...")
    result = calculate_metrics(0.0, 1.0)
    if result:
        results.append(result)
    
    # Run disease-specific analysis
    diseases = disease_df['disease'].unique()
    
    for disease in tqdm(diseases, desc="Evaluating disease-specific performance"):
        # With optimal balance for specific disease
        result = calculate_metrics(0.6, 0.4, disease)
        if result:
            results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs('results/comparative_analysis', exist_ok=True)
    results_df.to_csv('results/comparative_analysis/comparison_results.csv', index=False)
    
    # Generate visualizations
    # 1. Compare different weight configurations
    weight_configs = results_df[results_df['disease'] == 'all']
    
    if not weight_configs.empty:
        labels = [f"Emb:{row['embedding_weight']:.1f}, Imp:{row['importance_weight']:.1f}" 
                 for _, row in weight_configs.iterrows()]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.bar(labels, weight_configs['MRR'])
        plt.title('MRR by Weight Configuration')
        plt.xticks(rotation=45)
        plt.ylabel('MRR')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.bar(labels, weight_configs['Mean_Normalized_Rank'])
        plt.title('Mean Normalized Rank by Weight Configuration')
        plt.xticks(rotation=45)
        plt.ylabel('Mean Normalized Rank')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.subplot(2, 1, 2)
        
        hit_metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'Hit@10', 'Hit@20']
        for metric in hit_metrics:
            plt.plot(labels, weight_configs[metric], marker='o', label=metric)
        
        plt.title('Hit@k by Weight Configuration')
        plt.xticks(rotation=45)
        plt.ylabel('Hit Rate')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/comparative_analysis/weight_comparison.png', dpi=300)
    
    # 2. Compare performance across diseases
    disease_results = results_df[(results_df['disease'] != 'all') & 
                                (results_df['embedding_weight'] == 0.6) & 
                                (results_df['importance_weight'] == 0.4)]
    
    if not disease_results.empty:
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 2, 1)
        sns.barplot(x='disease', y='MRR', data=disease_results)
        plt.title('MRR by Disease')
        plt.xticks(rotation=45)
        plt.ylabel('MRR')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.subplot(2, 2, 2)
        sns.barplot(x='disease', y='Mean_Normalized_Rank', data=disease_results)
        plt.title('Mean Normalized Rank by Disease')
        plt.xticks(rotation=45)
        plt.ylabel('Mean Normalized Rank')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.subplot(2, 1, 2)
        disease_hit_data = pd.melt(disease_results, 
                                   id_vars=['disease'],
                                   value_vars=['Hit@1', 'Hit@5', 'Hit@10', 'Hit@20'],
                                   var_name='Metric', value_name='Value')
        
        sns.barplot(x='disease', y='Value', hue='Metric', data=disease_hit_data)
        plt.title('Hit@k by Disease')
        plt.xticks(rotation=45)
        plt.ylabel('Hit Rate')
        plt.legend(title='')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/comparative_analysis/disease_comparison.png', dpi=300)
    
    print("Comparative analysis completed. Results saved to results/comparative_analysis/ directory.")

if __name__ == "__main__":
    run_comparative_analysis()
