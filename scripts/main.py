import sys
import os
# Add the project root directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Now import modules from src
from src.data.kg_processor import KGProcessor
# ... rest of your imports
import torch
import pandas as pd
import numpy as np
import argparse
from torch_geometric.data import Data
import json
from tqdm import tqdm

from src.data.kg_processor import KGProcessor
from src.data.database_processor import DatabaseProcessor
from src.data.disease_importance_processor import DiseaseImportanceProcessor
from src.data.validated_processor import ValidatedInteractionsProcessor
from src.features.feature_builder import FeatureBuilder
from src.models.graph_sage import ImprovedGraphSAGE
from src.training.trainer import ModelTrainer
from src.evaluation.tcm_target_similarity import compute_all_tcm_targets
from src.evaluation.metrics import calculate_hit_at_k, calculate_mrr
from src.utils.visualization import visualize_results, visualize_compound_target_network

def prepare_data(args):
    """
    Prepare data and build graph.
    """
    print("Loading data...")
    
    # Load knowledge graph data
    kg_processor = KGProcessor(args.kg_data)
    
    # Load database data
    db_processor = DatabaseProcessor(args.db_data)
    
    # Load disease importance data
    disease_importance_processor = DiseaseImportanceProcessor(args.disease_importance)
    
    # Load validated interactions (if specified)
    validated_processor = None
    if args.validated_data:
        validated_processor = ValidatedInteractionsProcessor(args.validated_data)
    
    # Load cross-validation data if specified
    cv_train_data = None
    cv_test_data = None
    if args.cv_train_file and args.cv_test_file:
        print(f"Loading cross-validation data for fold {args.cv_fold}...")
        cv_train_data = pd.read_csv(args.cv_train_file)
        cv_test_data = pd.read_csv(args.cv_test_file)
    
    print("Building graph...")
    # Get all edges
    edges = kg_processor.get_edges() + db_processor.get_edges() + disease_importance_processor.get_edges()
    if validated_processor:
        edges += validated_processor.get_edges()
    
    # Create node map
    node_names = set()
    for src, dst, _, _ in edges:
        node_names.add(src)
        node_names.add(dst)
    
    node_map = {name: i for i, name in enumerate(node_names)}
    reverse_node_map = {i: name for name, i in node_map.items()}
    
    # Create edge index and attributes
    edge_index = []
    edge_attr = []
    for src, dst, rel_type, weight in edges:
        # Add undirected edges (both directions)
        edge_index.append([node_map[src], node_map[dst]])
        edge_index.append([node_map[dst], node_map[src]])
        
        # Edge attributes (relation type, weight)
        edge_attr.append([rel_type, weight])
        edge_attr.append([rel_type, weight])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Build node features
    feature_builder = FeatureBuilder(
        kg_processor, 
        db_processor, 
        disease_importance_processor, 
        feature_dim=args.feature_dim,
        validated_processor=validated_processor
    )
    
    node_features = feature_builder.build_features(node_map)
    
    # Create PyTorch Geometric data object
    data = Data(x=node_features, edge_index=edge_index)
    
    # Get compound indices
    compound_indices = []
    for name, idx in node_map.items():
        if (kg_processor.node_types.get(name) == 'compound' or 
            db_processor.node_types.get(name) == 'compound' or 
            (validated_processor and validated_processor.node_types.get(name) == 'compound')):
            compound_indices.append(idx)
    
    # Get target indices
    target_indices = []
    for name, idx in node_map.items():
        if (kg_processor.node_types.get(name) == 'target' or 
            db_processor.node_types.get(name) == 'target' or 
            (validated_processor and validated_processor.node_types.get(name) == 'target')):
            target_indices.append(idx)
    
    # Get disease importance targets
    important_targets = []
    for target_idx in target_indices:
        target_name = reverse_node_map[target_idx]
        if target_name in disease_importance_processor.get_feature_data()['target_importance']:
            important_targets.append(target_idx)
    
    # Get validated pairs for cross-validation
    validated_pairs = []
    if cv_train_data is not None:
        for _, row in cv_train_data.iterrows():
            if row['compound'] in node_map and row['target'] in node_map:
                compound_idx = node_map[row['compound']]
                target_idx = node_map[row['target']]
                validated_pairs.append((compound_idx, target_idx))
    
    # Test pairs for evaluation
    test_pairs = []
    if cv_test_data is not None:
        for _, row in cv_test_data.iterrows():
            if row['compound'] in node_map and row['target'] in node_map:
                compound_idx = node_map[row['compound']]
                target_idx = node_map[row['target']]
                test_pairs.append((compound_idx, target_idx))
    
    # Save intermediate data
    os.makedirs('data/processed', exist_ok=True)
    
    # Save node mapping
    with open('data/processed/node_map.json', 'w') as f:
        json.dump({str(k): v for k, v in node_map.items()}, f)
    
    with open('data/processed/reverse_node_map.json', 'w') as f:
        json.dump({str(k): v for k, v in reverse_node_map.items()}, f)
    
    # Save indices
    with open('data/processed/compound_indices.json', 'w') as f:
        json.dump(compound_indices, f)
    
    with open('data/processed/target_indices.json', 'w') as f:
        json.dump(target_indices, f)
    
    with open('data/processed/important_targets.json', 'w') as f:
        json.dump(important_targets, f)
    
    return data, node_map, reverse_node_map, compound_indices, target_indices, important_targets, validated_pairs, test_pairs


def train_model(data, args, important_targets=None, validated_pairs=None):
    """
    Train GraphSAGE model.
    """
    print("Training model...")
    
    # Initialize model
    model = ImprovedGraphSAGE(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        use_lstm_aggregator=args.use_lstm_aggregator
    )
    
    # Configure training parameters
    config = {
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'margin': args.margin,
        'early_stopping_patience': args.patience,
        'use_cuda': args.use_cuda
    }
    
    # Initialize trainer
    trainer = ModelTrainer(model, config)
    
    # Extract validated compounds and targets from pairs
    validated_compounds = []
    validated_targets = []
    if validated_pairs:
        for compound_idx, target_idx in validated_pairs:
            if compound_idx not in validated_compounds:
                validated_compounds.append(compound_idx)
            if target_idx not in validated_targets:
                validated_targets.append(target_idx)
    
    # Train model and get embeddings
    if validated_compounds and validated_targets:
        embeddings = trainer.train(data, validated_targets, validated_compounds)
    elif important_targets:
        embeddings = trainer.train(data, important_targets)
    else:
        embeddings = trainer.train(data)
    
    # Save embeddings
    os.makedirs('results/embeddings', exist_ok=True)
    torch.save(embeddings, 'results/embeddings/node_embeddings.pt')
    
    # Save model
    os.makedirs('results/models', exist_ok=True)
    torch.save(model.state_dict(), 'results/models/graphsage_model.pt')
    
    return embeddings


def prioritize_targets(embeddings, compound_indices, target_indices, edge_index, important_targets, reverse_node_map, test_pairs=None):
    """
    Prioritize targets for each compound.
    """
    print("Calculating target priorities...")
    
    # Create disease importance score mapping
    disease_importance = {}
    for target_idx in important_targets:
        # Give important targets a base score
        disease_importance[target_idx] = 0.8
    
    # Calculate target priorities for all compounds
    all_results = compute_all_tcm_targets(
        embeddings, 
        compound_indices, 
        target_indices, 
        disease_importance, 
        edge_index,
        embedding_weight=0.6,
        importance_weight=0.4
    )
    
    # Save results
    os.makedirs('results/prioritization', exist_ok=True)
    
    results_df_list = []
    for compound_idx, prioritized_targets in all_results.items():
        compound_name = reverse_node_map[compound_idx]
        
        for rank, (target_idx, score) in enumerate(prioritized_targets[:30], 1):
            target_name = reverse_node_map[target_idx]
            results_df_list.append({
                'compound': compound_name,
                'target': target_name,
                'score': score,
                'rank': rank,
                'is_important': target_idx in important_targets
            })
    
    results_df = pd.DataFrame(results_df_list)
    results_df.to_csv('results/prioritization/target_prioritization.csv', index=False)
    
    # Evaluate against test pairs if available
    if test_pairs:
        print("Evaluating against test pairs...")
        metrics_list = []
        mrr_values = []
        hit_values = {k: [] for k in [1, 3, 5, 10, 20]}
        
        for compound_idx, target_idx in test_pairs:
            if compound_idx in all_results:
                prioritized_targets = all_results[compound_idx]
                validation_set = [target_idx]
                
                # Calculate metrics
                hit_metrics = calculate_hit_at_k(prioritized_targets, validation_set)
                mrr = calculate_mrr(prioritized_targets, validation_set)
                
                mrr_values.append(mrr)
                for k in hit_values.keys():
                    hit_values[k].append(hit_metrics[f'Hit@{k}'])
                
                # Determine rank
                rank = -1
                for i, (t_idx, _) in enumerate(prioritized_targets):
                    if t_idx == target_idx:
                        rank = i + 1
                        break
                
                if rank != -1:
                    metrics_list.append({
                        'compound': reverse_node_map[compound_idx],
                        'target': reverse_node_map[target_idx],
                        'rank': rank,
                        'mrr': mrr
                    })
        
        # Save detailed metrics
        if metrics_list:
            pd.DataFrame(metrics_list).to_csv('results/evaluation/detailed_metrics.csv', index=False)
            
            # Calculate summary metrics
            avg_mrr = np.mean(mrr_values)
            mean_rank = np.mean([m['rank'] for m in metrics_list])
            mean_normalized_rank = mean_rank / len(target_indices)
            hit_rates = {k: np.mean(hit_values[k]) for k in hit_values.keys()}
            
            # Save summary metrics
            metrics_df = pd.DataFrame({
                'MRR': [avg_mrr],
                'Mean Rank': [mean_rank],
                'Mean Normalized Rank': [mean_normalized_rank],
                'Hit@1': [hit_rates[1]],
                'Hit@3': [hit_rates[3]],
                'Hit@5': [hit_rates[5]],
                'Hit@10': [hit_rates[10]],
                'Hit@20': [hit_rates[20]]
            })
            
            # Create results directory
            os.makedirs(f'results/{"cv_fold_" + str(args.cv_fold) if args.cv_fold else "evaluation"}', exist_ok=True)
            metrics_df.to_csv(f'results/{"cv_fold_" + str(args.cv_fold) if args.cv_fold else "evaluation"}/metrics.csv', index=False)
            
            print("\nEvaluation Metrics:")
            print(f"MRR: {avg_mrr:.4f}")
            print(f"Mean Rank: {mean_rank:.2f}")
            print(f"Mean Normalized Rank: {mean_normalized_rank:.4f}")
            for k in sorted(hit_rates.keys()):
                print(f"Hit@{k}: {hit_rates[k]:.4f}")
    
    return all_results


def main(args):
    # Prepare data
    data, node_map, reverse_node_map, compound_indices, target_indices, important_targets, validated_pairs, test_pairs = prepare_data(args)
    
    # Train model
    embeddings = train_model(data, args, important_targets, validated_pairs)
    
    # Prioritize targets
    all_results = prioritize_targets(embeddings, compound_indices, target_indices, data.edge_index, important_targets, reverse_node_map, test_pairs)
    
    # Visualize first compound's results as example
    if compound_indices and not args.cv_fold:
        first_compound_idx = compound_indices[0]
        first_compound_targets = all_results[first_compound_idx]
        
        print(f"Example: Compound {reverse_node_map[first_compound_idx]} target prioritization")
        for rank, (target_idx, score) in enumerate(first_compound_targets[:10], 1):
            print(f"  {rank}. {reverse_node_map[target_idx]}: {score:.4f}")
        
        # Visualize results
        visualize_results(
            embeddings,
            node_map,
            reverse_node_map,
            compound_indices[:5],  # Only show first 5 compounds
            first_compound_targets,
            important_targets
        )
        
        # Visualize network
        visualize_compound_target_network(
            data.edge_index,
            node_map,
            reverse_node_map,
            compound_indices[:5],  # Only show first 5 compounds
            first_compound_targets,
            important_targets
        )
    
    print("TCM target prioritization complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCM Target Prioritization System")
    
    # Data parameters
    parser.add_argument('--kg_data', type=str, default='data/raw/kg_data_extended.csv',
                        help='Knowledge graph data file path')
    parser.add_argument('--db_data', type=str, default='data/raw/database_data_extended.csv',
                        help='Database data file path')
    parser.add_argument('--disease_importance', type=str, default='data/raw/disease_importance_extended.csv',
                        help='Disease target importance data file path')
    parser.add_argument('--validated_data', type=str, default='',
                        help='Validated interactions data file path')
    
    # Cross-validation parameters
    parser.add_argument('--cv_fold', type=int, default=None,
                        help='Cross-validation fold number')
    parser.add_argument('--cv_train_file', type=str, default='',
                        help='Cross-validation training data file')
    parser.add_argument('--cv_test_file', type=str, default='',
                        help='Cross-validation test data file')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='Output embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--use_lstm_aggregator', action='store_true',
                        help='Use LSTM aggregator')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA if available')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--margin', type=float, default=0.4,
                        help='Contrastive loss margin')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)
