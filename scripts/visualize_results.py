import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import json
import torch
from sklearn.manifold import TSNE

def visualize_metrics():
    """Visualize model evaluation metrics"""
    # Check if metrics file exists
    metrics_file = 'results/evaluation/validated_metrics.csv'
    if not os.path.exists(metrics_file):
        print("Metrics file not found. Please run evaluation first.")
        return
    
    # Load metrics
    metrics_df = pd.read_csv(metrics_file)
    
    # Create directory for visualizations
    os.makedirs('results/visualizations', exist_ok=True)
    
    # 1. Hit@k bar chart
    plt.figure(figsize=(12, 6))
    hit_metrics = ['Hit@1', 'Hit@3', 'Hit@5', 'Hit@10', 'Hit@20']
    hit_values = metrics_df[hit_metrics].values[0]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = plt.bar(hit_metrics, hit_values, color=colors)
    
    plt.title('Hit@k Metrics', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/hit_at_k_metrics.png', dpi=300)
    
    # 2. MRR and Mean Normalized Rank
    plt.figure(figsize=(10, 6))
    rank_metrics = ['MRR', 'Mean Normalized Rank']
    rank_values = metrics_df[rank_metrics].values[0]
    
    plt.bar(rank_metrics, rank_values, color=['#1f77b4', '#ff7f0e'])
    plt.title('Ranking Metrics', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add value labels
    for i, v in enumerate(rank_values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/ranking_metrics.png', dpi=300)
    
    # If cross-validation results exist, visualize them too
    cv_file = 'results/cross_validation/cv_metrics.csv'
    if os.path.exists(cv_file):
        cv_df = pd.read_csv(cv_file)
        
        # 3. Cross-validation Hit@k boxplot
        plt.figure(figsize=(14, 8))
        hit_data = cv_df[hit_metrics]
        
        # Create a boxplot
        sns.boxplot(data=hit_data)
        plt.title('Cross-Validation Hit@k Performance', fontsize=16)
        plt.xlabel('Metric', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/cv_hit_at_k_boxplot.png', dpi=300)
        
        # 4. Cross-validation MRR and Mean Normalized Rank boxplot
        plt.figure(figsize=(10, 6))
        rank_data = cv_df[rank_metrics]
        
        sns.boxplot(data=rank_data)
        plt.title('Cross-Validation Ranking Metrics', fontsize=16)
        plt.xlabel('Metric', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/cv_ranking_boxplot.png', dpi=300)
    
    print("Visualizations created in results/visualizations/ directory")

def visualize_target_rankings():
    """Visualize rankings of specific targets for each compound"""
    # Check if rankings file exists
    rankings_file = 'results/evaluation/validated_rankings.csv'
    if not os.path.exists(rankings_file):
        print("Rankings file not found. Please run evaluation first.")
        return
    
    # Load rankings
    rankings_df = pd.read_csv(rankings_file)
    
    # Create directory for visualizations
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Group by compound and calculate statistics
    compound_stats = rankings_df.groupby('Compound')['Rank'].agg(['mean', 'min', 'max', 'count']).reset_index()
    compound_stats = compound_stats.sort_values('mean')
    
    # 1. Compound average rank bar chart
    plt.figure(figsize=(14, 8))
    bars = plt.bar(compound_stats['Compound'], compound_stats['mean'], yerr=compound_stats['max']-compound_stats['min'])
    plt.title('Average Target Rank by Compound', fontsize=16)
    plt.xlabel('Compound', fontsize=14)
    plt.ylabel('Average Rank', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add count labels
    for bar, count in zip(bars, compound_stats['count']):
        plt.text(bar.get_x() + bar.get_width()/2., 0.5, 
                f'n={count}', ha='center', va='bottom', rotation=90, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/compound_average_rank.png', dpi=300)
    
    # 2. Detailed rankings heatmap (for compounds with multiple targets)
    compounds_with_multiple = compound_stats[compound_stats['count'] > 1]['Compound'].tolist()
    
    if compounds_with_multiple:
        # Filter to compounds with multiple targets
        multi_df = rankings_df[rankings_df['Compound'].isin(compounds_with_multiple)]
        
        # Pivot to create a compound x target matrix
        pivot_df = multi_df.pivot(index='Compound', columns='Target', values='Rank')
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu_r', fmt='.0f', linewidths=0.5)
        plt.title('Target Ranks by Compound', fontsize=16)
        plt.ylabel('Compound', fontsize=14)
        plt.xlabel('Target', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/compound_target_rank_heatmap.png', dpi=300)
    
    print("Target ranking visualizations created in results/visualizations/ directory")

def analyze_embeddings():
    """Analyze and visualize the learned embeddings"""
    # Check if embeddings file exists
    embeddings_file = 'results/embeddings/node_embeddings.pt'
    node_map_file = 'data/processed/node_map.json'
    
    if not (os.path.exists(embeddings_file) and os.path.exists(node_map_file)):
        print("Embeddings or node mapping file not found.")
        return
    
    # Load embeddings and node mapping
    embeddings = torch.load(embeddings_file)
    
    with open(node_map_file, 'r') as f:
        node_map = json.load(f)
    
    # Create directory for visualizations
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Perform dimensionality reduction for visualization
    # Convert embeddings to numpy for TSNE
    embeddings_np = embeddings.numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Create a dataframe for the 2D embeddings
    node_types = []
    node_names = []
    
    for name, idx in node_map.items():
        idx = int(idx)  # Ensure index is integer
        
        # Determine node type based on name
        if isinstance(name, str):
            if name.startswith(('IL', 'TNF', 'CASP', 'MMP', 'BCL', 'P53', 'SOD')):
                node_type = 'Target'
            elif any(name.startswith(p) for p in ['Berberine', 'Curcumin', 'Ginsenoside', 'Baicalein']):
                node_type = 'Compound'
            elif any(name.startswith(d) for d in ['Silicosis', 'Liver_Fibrosis', 'IBD', 'Diabetes']):
                node_type = 'Disease'
            else:
                node_type = 'Other'
        else:
            node_type = 'Unknown'
        
        node_types.append(node_type)
        node_names.append(name)
    
    # Create dataframe
    tsne_df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'name': node_names,
        'type': node_types
    })
    
    # Plot t-SNE visualization
    plt.figure(figsize=(12, 10))
    
    # Plot by node type with different colors
    for node_type, color in zip(['Compound', 'Target', 'Disease', 'Other'], 
                               ['#ff7f0e', '#1f77b4', '#2ca02c', '#d3d3d3']):
        subset = tsne_df[tsne_df['type'] == node_type]
        plt.scatter(subset['x'], subset['y'], c=color, label=node_type, alpha=0.7)
    
    # Annotate a few key nodes
    compounds = tsne_df[tsne_df['type'] == 'Compound'].sample(min(5, len(tsne_df[tsne_df['type'] == 'Compound'])))
    targets = tsne_df[tsne_df['type'] == 'Target'].sample(min(5, len(tsne_df[tsne_df['type'] == 'Target'])))
    diseases = tsne_df[tsne_df['type'] == 'Disease']
    
    for _, row in pd.concat([compounds, targets, diseases]).iterrows():
        plt.annotate(row['name'], (row['x'], row['y']), fontsize=8)
    
    plt.title('t-SNE Visualization of Node Embeddings', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/tsne_embeddings.png', dpi=300)
    
    print("Embedding visualizations created in results/visualizations/ directory")

if __name__ == "__main__":
    print("Generating visualizations...")
    visualize_metrics()
    visualize_target_rankings()
    analyze_embeddings()
    print("All visualizations completed!")
