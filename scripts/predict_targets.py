import pandas as pd
import numpy as np
import torch
import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def predict_targets(args):
    """Predict and rank potential targets for a specific compound and disease"""
    print(f"Predicting potential targets for {args.compound} in the context of {args.disease}...")
    
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
    
    # Check if compound exists
    if args.compound not in node_map:
        print(f"Error: Compound '{args.compound}' not found in the node map.")
        print("Available compounds:")
        compounds = [name for name in node_map.keys() 
                    if isinstance(name, str) and any(name.startswith(p) 
                                                   for p in ['Berberine', 'Curcumin', 'Ginsenoside', 'Baicalein', 
                                                            'Astragaloside', 'Quercetin', 'Tanshinone', 'Tetrandrine',
                                                            'Emodin', 'Resveratrol', 'Huperzine', 'Ligustrazine'])]
        for compound in sorted(compounds):
            print(f"  - {compound}")
        return
    
    # Load disease importance data
    disease_file = 'data/raw/disease_importance_extended.csv'
    if not os.path.exists(disease_file):
        print("Disease importance data not found.")
        return
    
    disease_df = pd.read_csv(disease_file)
    
    # Filter by disease if specified
    if args.disease and args.disease != 'all':
        if args.disease not in disease_df['disease'].unique():
            print(f"Error: Disease '{args.disease}' not found in the data.")
            print("Available diseases:")
            for disease in sorted(disease_df['disease'].unique()):
                print(f"  - {disease}")
            return
        
        disease_df = disease_df[disease_df['disease'] == args.disease]
    
    # Create disease importance dictionary
    disease_importance = {}
    
    for _, row in disease_df.iterrows():
        target = row['target']
        importance = row['importance_score']
        
        if target in node_map:
            target_idx = node_map[target]
            disease_importance[target_idx] = importance
    
    # Get all targets
    target_nodes = [name for name, idx in node_map.items() 
                   if isinstance(name, str) and any(name.startswith(prefix) 
                                                  for prefix in ['Target_', 'IL', 'TNF', 'CASP', 'MMP', 'BCL', 'P',
                                                                'SOD', 'CAT', 'GPX', 'NOS', 'AKT', 'MAPK', 'JAK',
                                                                'STAT', 'NFKB', 'PTGS', 'VEGF', 'FGF', 'IGF'])]
    
    # Get compound embedding
    compound_idx = node_map[args.compound]
    compound_emb = embeddings[compound_idx].unsqueeze(0)
    
    # Calculate similarity with all targets
    similarities = []
    for target_name in target_nodes:
        if target_name in node_map:
            target_idx = node_map[target_name]
            target_emb = embeddings[target_idx].unsqueeze(0)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                compound_emb, target_emb, dim=1).item()
            
            # Get target importance
            importance_score = disease_importance.get(target_idx, 0.0)
            
            # Calculate combined score with configurable weights
            combined_score = (
                args.embedding_weight * similarity +
                args.importance_weight * importance_score
            )
            
            # Add to list
            similarities.append({
                'target': target_name,
                'similarity': similarity,
                'importance': importance_score,
                'combined_score': combined_score
            })
    
    # Convert to DataFrame and sort by combined score
    results_df = pd.DataFrame(similarities)
    results_df = results_df.sort_values('combined_score', ascending=False)
    
    # Create directory for results
    os.makedirs('results/predictions', exist_ok=True)
    
    # Save detailed results
    output_file = f'results/predictions/{args.compound}_{args.disease}_targets.csv'
    results_df.to_csv(output_file, index=False)
    
    # Print top targets
    print(f"\nTop {args.top_k} predicted targets for {args.compound} in {args.disease}:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Target':<15} {'Combined Score':<15} {'Similarity':<15} {'Importance':<15}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(results_df.head(args.top_k).iterrows()):
        print(f"{i+1:<5} {row['target']:<15} {row['combined_score']:.4f}{' ':8} {row['similarity']:.4f}{' ':8} {row['importance']:.4f}")
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Top targets bar chart
    top_k_results = results_df.head(args.top_k)
    
    plt.subplot(2, 1, 1)
    bars = plt.barh(top_k_results['target'], top_k_results['combined_score'])
    plt.title(f'Top {args.top_k} Predicted Targets for {args.compound} in {args.disease}')
    plt.xlabel('Combined Score')
    plt.ylabel('Target')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{top_k_results.iloc[i]["combined_score"]:.4f}', 
                va='center')
    
    # Component contribution chart
    plt.subplot(2, 1, 2)
    
    # Create a stacked bar chart
    width = 0.4
    ind = np.arange(len(top_k_results))
    
    # Scale components
    sim_component = top_k_results['similarity'] * args.embedding_weight
    imp_component = top_k_results['importance'] * args.importance_weight
    
    plt.barh(ind, sim_component, width, label='Embedding Similarity Component')
    plt.barh(ind, imp_component, width, left=sim_component, label='Disease Importance Component')
    
    plt.yticks(ind, top_k_results['target'])
    plt.title('Component Contribution to Combined Score')
    plt.xlabel('Component Weight')
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'results/predictions/{args.compound}_{args.disease}_targets.png', dpi=300)
    
    print(f"\nResults saved to {output_file}")
    print(f"Visualization saved to results/predictions/{args.compound}_{args.disease}_targets.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict potential targets for a compound in a disease context")
    
    parser.add_argument('--compound', type=str, required=True,
                      help='Compound name')
    parser.add_argument('--disease', type=str, default='all',
                      help='Disease name (default: all)')
    parser.add_argument('--top_k', type=int, default=10,
                      help='Number of top targets to show (default: 10)')
    parser.add_argument('--embedding_weight', type=float, default=0.6,
                      help='Weight for embedding similarity (default: 0.6)')
    parser.add_argument('--importance_weight', type=float, default=0.4,
                      help='Weight for disease importance (default: 0.4)')
    
    args = parser.parse_args()
    predict_targets(args)
