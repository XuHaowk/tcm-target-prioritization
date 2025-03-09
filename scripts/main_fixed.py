#!/usr/bin/env python3
"""
Enhanced TCM Target Prioritization Main Script
"""

#!/usr/bin/env python3
"""
Enhanced TCM Target Prioritization Main Script
This script integrates graph neural networks and disease importance
for Traditional Chinese Medicine compound-target prioritization.
"""

import os
import sys
# 添加项目根目录到Python的路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 导入自定义模块
from src.data.kg_processor import load_knowledge_graph
from src.data.database_processor import load_database
from src.data.disease_importance_processor import load_disease_importance
from src.data.validated_data_processor import load_validated_interactions
from src.features.feature_builder import build_node_features, enhance_node_features, normalize_features
from src.data.graph_builder import build_graph
from src.models.graph_sage import ImprovedGraphSAGE
from src.models.aggregators import LSTMAggregator, AttentionAggregator
from src.training.trainer import Trainer
from src.evaluation.tcm_target_similarity import compute_all_tcm_targets
from src.utils.visualization import visualize_embeddings, create_priority_heatmap

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TCM Target Prioritization')
    
    # Data arguments
    parser.add_argument('--kg_data', type=str, required=True, help='Knowledge graph data file')
    parser.add_argument('--db_data', type=str, required=True, help='Database data file')
    parser.add_argument('--disease_importance', type=str, required=True, help='Disease importance file')
    parser.add_argument('--validated_data', type=str, required=False, help='Validated interactions file')
    
    # Model arguments
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=64, help='Output dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--use_lstm_aggregator', action='store_true', help='Use LSTM aggregator')
    parser.add_argument('--model_type', type=str, default='graphsage', choices=['graphsage', 'gat'], 
                        help='GNN model type (graphsage or gat)')
    parser.add_argument('--attention_heads', type=int, default=8, help='Number of attention heads for GAT')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--margin', type=float, default=0.3, help='Margin for loss function')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda)')
    
    # Evaluation arguments
    parser.add_argument('--embedding_weight', type=float, default=0.6, help='Weight for embedding similarity')
    parser.add_argument('--importance_weight', type=float, default=0.4, help='Weight for target importance')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top targets to return')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--save_model', action='store_true', help='Save model')
    parser.add_argument('--save_embeddings', action='store_true', help='Save embeddings')
    parser.add_argument('--save_priorities', action='store_true', help='Save priorities')
    parser.add_argument('--visualize', action='store_true', help='Visualize embeddings')
    
    return parser.parse_args()

def load_data(args):
    """
    Load and process all required data
    
    Args:
        args: Command line arguments
        
    Returns:
        Processed data for model training and evaluation
    """
    print("Loading data...")
    
    # Load knowledge graph data
    kg_data = load_knowledge_graph(args.kg_data)
    print(f"Successfully loaded data: {args.kg_data}")
    
    # Load database data
    db_data = load_database(args.db_data)
    print(f"Successfully loaded data: {args.db_data}")
    
    # Load disease importance data
    disease_importance = load_disease_importance(args.disease_importance)
    print(f"Successfully loaded data: {args.disease_importance}")
    
    # Load validated interactions if available
    validated_pairs = None
    if args.validated_data:
        validated_pairs = load_validated_interactions(args.validated_data)
        print(f"Successfully loaded data: {args.validated_data}")
    
    # Extract compound-target interactions from knowledge graph
    compound_target_data = kg_data[kg_data['relation_type'] == 'compound_target']
    
    # Extract disease-target interactions from knowledge graph
    disease_target_data = kg_data[kg_data['relation_type'] == 'disease_target']
    
    # Extract unique compounds and targets
    compounds = list(set(compound_target_data['compound_id']))
    targets = list(set(compound_target_data['target_id']))
    diseases = list(set(disease_target_data['disease_id']))
    
    # Get compound features from database
    compound_features = {}
    for _, row in db_data.iterrows():
        compound_id = row['compound_id']
        features = np.array(eval(row['feature_vector']))
        compound_features[compound_id] = torch.tensor(features, dtype=torch.float32)
    
    # Create simple target features (could be enhanced with external data)
    target_features = {}
    for target_id in targets:
        # For now, using random features - ideally replaced with protein features
        features = torch.randn(args.feature_dim)
        target_features[target_id] = features
    
    # Create disease features
    disease_features = {}
    for disease_id in diseases:
        # For now, using random features
        features = torch.randn(args.feature_dim)
        disease_features[disease_id] = features
    
    # Create important targets dictionary for prioritization
    important_targets = {}
    for _, row in disease_importance.iterrows():
        target_id = row['target_id']
        importance = row['importance_score']
        important_targets[target_id] = float(importance)
    
    return (compound_target_data, disease_target_data, compound_features, 
            target_features, disease_features, important_targets, validated_pairs)

def train_model(data, args, important_targets, validated_pairs=None):
    """
    Train GNN model on graph data
    
    Args:
        data: Graph Data object
        args: Command line arguments
        important_targets: Dictionary of target importances
        validated_pairs: Validated compound-target pairs (optional)
        
    Returns:
        Node embeddings
    """
    # Create the model based on arguments
    if args.model_type == 'gat':
        try:
            from src.models.gat import ImprovedGAT
            model = ImprovedGAT(
                in_dim=data.x.shape[1],
                hidden_dim=args.hidden_dim, 
                out_dim=args.output_dim,
                heads=args.attention_heads,
                dropout=args.dropout
            )
            print("Using GAT model with attention mechanism")
        except ImportError:
            print("GAT model not available, falling back to GraphSAGE")
            model = ImprovedGraphSAGE(
                in_dim=data.x.shape[1],
                hidden_dim=args.hidden_dim,
                out_dim=args.output_dim,
                dropout=args.dropout
            )
    elif args.use_lstm_aggregator:
        aggregator = LSTMAggregator(args.hidden_dim)
        model = ImprovedGraphSAGE(
            in_dim=data.x.shape[1],
            hidden_dim=args.hidden_dim,
            out_dim=args.output_dim,
            dropout=args.dropout,
            aggregator=aggregator
        )
        print("Using GraphSAGE model with LSTM aggregator")
    else:
        model = ImprovedGraphSAGE(
            in_dim=data.x.shape[1],
            hidden_dim=args.hidden_dim,
            out_dim=args.output_dim,
            dropout=args.dropout
        )
        print("Using basic GraphSAGE model")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        margin=args.margin,
        patience=args.patience,
        device=args.device
    )
    
    # Train the model
    embeddings = trainer.train(data, important_targets)
    
    # Save the model if requested
    if args.save_model:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f"model_{args.model_type}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Save embeddings if requested
    if args.save_embeddings:
        os.makedirs(args.output_dir, exist_ok=True)
        embeddings_path = os.path.join(args.output_dir, "embeddings.pt")
        torch.save(embeddings, embeddings_path)
        print(f"Embeddings saved to {embeddings_path}")
    
    return embeddings

def print_example_targets(all_priorities, reverse_node_map, compound_id=None):
    """
    Print example compound target priorities
    
    Args:
        all_priorities: Dictionary of compound priorities
        reverse_node_map: Mapping from indices to node IDs
        compound_id: Specific compound ID to print (optional)
    """
    # If no compound specified, use the first one
    if compound_id is None and all_priorities:
        compound_id = next(iter(all_priorities.keys()))
    
    if compound_id in all_priorities:
        print(f"Example: Compound {compound_id} target prioritization")
        for rank, (target_idx, score) in enumerate(all_priorities[compound_id][:10], 1):
            # Get target name with error handling
            if target_idx in reverse_node_map:
                target_name = reverse_node_map[target_idx]
                # Check if target name is valid
                if isinstance(target_name, float) and np.isnan(target_name):
                    target_name = f"Target_{target_idx}"
            else:
                target_name = f"Target_{target_idx}"
                
            print(f"  {rank}. {target_name}: {score:.4f}")

def main(args):
    """
    Main function for TCM target prioritization
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all required data
    data_result = load_data(args)
    (compound_target_data, disease_target_data, compound_features, 
     target_features, disease_features, important_targets, valided_pairs) = data_result
    
    # Build graph with enhanced features
    data, node_map, reverse_node_map = build_graph(
        compound_target_data,
        disease_target_data,
        compound_features,
        target_features,
        disease_features
    )
    
    # Train model and get embeddings
    embeddings = train_model(data, args, important_targets, valided_pairs)
    
    # Compute target priorities for all compounds
    all_priorities = compute_all_tcm_targets(
        embeddings, 
        node_map, 
        reverse_node_map, 
        important_targets, 
        data.compound_indices, 
        data.target_indices,
        validated_pairs=valided_pairs  # Add validated data for dynamic weight training
    )
    
    # Print example results
    print_example_targets(all_priorities, reverse_node_map)
    
    # Save priorities if requested
    if args.save_priorities:
        priorities_df = []
        for compound_id, targets in all_priorities.items():
            for target_idx, score in targets[:args.top_k]:
                target_id = reverse_node_map.get(target_idx, f"Target_{target_idx}")
                priorities_df.append({
                    'compound_id': compound_id,
                    'target_id': target_id,
                    'score': score,
                })
        
        priorities_df = pd.DataFrame(priorities_df)
        priorities_path = os.path.join(args.output_dir, "target_prioritization.csv")
        priorities_df.to_csv(priorities_path, index=False)
        print(f"Target priorities saved to {priorities_path}")
    
    # Visualize embeddings if requested
    if args.visualize:
        # Visualize only a subset for clarity
        max_nodes = 500
        if data.x.shape[0] > max_nodes:
            selected_indices = torch.randperm(data.x.shape[0])[:max_nodes]
        else:
            selected_indices = torch.arange(data.x.shape[0])
        
        # Handle potential NaN values in embeddings
        if torch.isnan(embeddings).any():
            print("Warning: NaN values detected in embeddings. Replacing with zeros for visualization.")
            embeddings = torch.nan_to_num(embeddings, nan=0.0)
        
        # Create visualization
        vis_path = os.path.join(args.output_dir, "embedding_visualization.png")
        visualize_embeddings(
            embeddings, 
            selected_indices, 
            data.compound_indices, 
            data.target_indices, 
            data.disease_indices if hasattr(data, 'disease_indices') else None,
            output_path=vis_path
        )
        print(f"Embedding visualization saved to {vis_path}")
        
        # Create priority heatmap for top compounds
        try:
            heatmap_path = os.path.join(args.output_dir, "priority_heatmap.png")
            create_priority_heatmap(
                all_priorities, 
                reverse_node_map, 
                num_compounds=10, 
                num_targets=20,
                output_path=heatmap_path
            )
            print(f"Priority heatmap saved to {heatmap_path}")
        except Exception as e:
            print(f"Could not create priority heatmap: {e}")
    
    print("TCM target prioritization completed!")
    return embeddings, all_priorities

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run main function
    main(args)
