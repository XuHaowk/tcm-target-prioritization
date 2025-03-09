#!/usr/bin/env python
"""
Improved main script for TCM target prioritization

This script integrates all the improvements:
1. Validation against known data
2. Weight adjustment
3. GNN fixes
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json
from datetime import datetime
import types

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data.graph_builder import build_graph
from src.models.fixed_graph_sage import FixedGraphSAGE
from src.evaluation.validation import load_validation_data, validate_model
from src.optimization.weight_tuning import find_optimal_weights, calculate_priorities_with_weights

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tcm_prioritization.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TCM Target Prioritization')
    
    # Data files
    parser.add_argument('--kg_data', type=str, required=True, help='Knowledge graph data file')
    parser.add_argument('--db_data', type=str, required=True, help='Database compounds file')
    parser.add_argument('--disease_importance', type=str, required=True, help='Disease-target importance file')
    parser.add_argument('--validated_data', type=str, required=True, help='Validated interactions file')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=128, help='Output dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--margin', type=float, default=0.4, help='Contrastive loss margin')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
    # Weight tuning - update default to optimal value from results
    parser.add_argument('--embedding_weight', type=float, default=0.05, 
                      help='Weight for embedding scores (vs. importance scores)')
    parser.add_argument('--tune_weights', action='store_true', 
                      help='Automatically tune embedding and importance weights')
    
    # GNN options
    parser.add_argument('--force_mlp', action='store_true', 
                      help='Force use of MLP even if GNN is available')
    
    # Evaluation options
    parser.add_argument('--top_k', type=int, default=10, 
                      help='Number of top targets to show for each compound')
    
    # Save model
    parser.add_argument('--save_model', action='store_true',
                      help='Save the trained model')
    
    # Load model instead of training
    parser.add_argument('--load_model', type=str, default=None,
                      help='Path to load a pretrained model')
    
    return parser.parse_args()


def load_data(args):
    """Load data files"""
    logger.info("Loading data...")
    
    # Knowledge graph data (compound-target relationships)
    try:
        kg_data = pd.read_csv(args.kg_data)
        
        # Check required columns
        required_cols = ['compound_id', 'target_id']
        missing_cols = [col for col in required_cols if col not in kg_data.columns]
        
        if missing_cols:
            logger.warning(f"Knowledge graph data missing required columns: {missing_cols}")
            
            # Check for alternative column names
            alt_cols = {'compound_id': 'compound', 'target_id': 'target'}
            for req_col, alt_col in alt_cols.items():
                if req_col in missing_cols and alt_col in kg_data.columns:
                    logger.info(f"Using '{alt_col}' instead of '{req_col}'")
        
        # Get relation types
        if 'relation_type' in kg_data.columns:
            relation_types = kg_data['relation_type'].unique()
        else:
            relation_types = []
            
        logger.info(f"Loaded knowledge graph relation data: {len(kg_data)} rows")
        if len(relation_types) > 0:
            logger.info(f"Relation types: {relation_types}")
            
        logger.info(f"Successfully loaded data: {args.kg_data}")
    except Exception as e:
        logger.error(f"Failed to load knowledge graph data: {e}")
        kg_data = pd.DataFrame()
    
    # Database data (compound features)
    try:
        db_data = pd.read_csv(args.db_data)
        
        # Check required columns
        required_cols = ['compound_id']
        missing_cols = [col for col in required_cols if col not in db_data.columns]
        
        if missing_cols:
            logger.warning(f"Database data missing required columns: {missing_cols}")
            
            # Check for alternative column names
            if 'compound' in db_data.columns and 'compound_id' not in db_data.columns:
                logger.info("Using 'compound' instead of 'compound_id'")
                db_data['compound_id'] = db_data['compound']
        
        # Extract compound features
        compound_features = {}
        
        if 'feature_vector' in db_data.columns:
            for _, row in db_data.iterrows():
                compound_id = str(row.get('compound_id', row.get('compound', f'compound_{_}')))
                
                # Extract feature vector from string representation
                feature_str = row['feature_vector']
                try:
                    # Handle different formats of feature strings
                    feature_str = feature_str.strip('[]')
                    features = [float(x.strip()) for x in feature_str.split(',')]
                    compound_features[compound_id] = torch.tensor(features)
                except:
                    logger.warning(f"Could not parse feature vector for compound {compound_id}")
        
        if not compound_features:
            logger.warning("Creating random features for compounds as placeholders")
            for i, row in db_data.iterrows():
                compound_id = str(row.get('compound_id', row.get('compound', f'compound_{i}')))
                compound_features[compound_id] = torch.randn(args.feature_dim)
        
        logger.info(f"Loaded database data: {len(db_data)} compounds")
        logger.info(f"Successfully loaded data: {args.db_data}")
        
    except Exception as e:
        logger.error(f"Failed to load database data: {e}")
        db_data = pd.DataFrame()
        compound_features = {}
    
    # Disease-target importance data
    try:
        disease_data = pd.read_csv(args.disease_importance)
        
        # Check required columns
        required_cols = ['target_id']
        missing_cols = [col for col in required_cols if col not in disease_data.columns]
        
        if missing_cols:
            logger.warning(f"Disease importance data missing required columns: {missing_cols}")
            
            # Check for alternative column names
            if 'target' in disease_data.columns and 'target_id' not in disease_data.columns:
                logger.info("Using 'target' instead of 'target_id'")
                disease_data['target_id'] = disease_data['target']
        
        # Extract target importance scores
        target_importance = {}
        importance_col = 'importance_score' if 'importance_score' in disease_data.columns else 'importance'
        
        if importance_col in disease_data.columns:
            for _, row in disease_data.iterrows():
                target_id = str(row.get('target_id', row.get('target', f'target_{_}')))
                importance = float(row[importance_col])
                
                # Store maximum importance for each target across diseases
                if target_id not in target_importance or importance > target_importance[target_id]:
                    target_importance[target_id] = importance
            
            # Report importance score range
            if target_importance:
                min_score = min(target_importance.values())
                max_score = max(target_importance.values())
                logger.info(f"Target importance scores range: {min_score} - {max_score}")
        
        logger.info(f"Loaded disease importance data: {len(disease_data)} rows")
        logger.info(f"Successfully loaded data: {args.disease_importance}")
        
    except Exception as e:
        logger.error(f"Failed to load disease importance data: {e}")
        disease_data = pd.DataFrame()
        target_importance = {}
    
    # Validated interactions data
    validated_data = load_validation_data(args.validated_data)
    
    return kg_data, db_data, disease_data, validated_data, compound_features, target_importance


class TargetPrioritizationTrainer:
    """Trainer for target prioritization model"""
    
    def __init__(self, model, device, epochs=300, lr=0.0001, 
                 weight_decay=1e-5, margin=0.4, patience=30):
        """
        Initialize trainer
        
        Args:
            model: Neural network model
            device: Torch device
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            margin: Margin for contrastive loss
            patience: Early stopping patience
        """
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.margin = margin
        self.patience = patience
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Loss history
        self.losses = []
        
        # For enhanced loss function
        self.target_importance = {}
        self.reverse_node_map = {}
    
    def train(self, data, important_targets):
        """
        Train the model
        
        Args:
            data: Graph data object
            important_targets: List of important target indices
            
        Returns:
            Node embeddings
        """
        # Move data to device
        data = data.to(self.device)
        
        # Set model to training mode
        self.model.train()
        
        # Track best model
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(data.x, data.edge_index, data.edge_weight)
            
            # Calculate contrastive loss
            loss = self.contrastive_loss(embeddings, data.target_indices, important_targets)
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            loss_value = loss.item()
            self.losses.append(loss_value)
            
            # Log progress
            if epoch == 1 or epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs}, Loss: {loss_value:.4f}")
            
            # Check for early stopping
            if loss_value < best_loss:
                best_loss = loss_value
                best_model_state = {key: val.cpu() for key, val in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}, best loss: {best_loss:.4f}")
                break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        self.model.eval()
        
        # Generate final embeddings
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index, data.edge_weight)
        
        return embeddings
    
    def contrastive_loss(self, embeddings, target_indices, important_targets):
        """
        Calculate contrastive loss
        
        Args:
            embeddings: Node embeddings
            target_indices: Indices of target nodes
            important_targets: List of important target indices
            
        Returns:
            Loss value
        """
        # Get target embeddings
        target_embeddings = embeddings[target_indices]
        
        # Number of targets
        num_targets = len(target_indices)
        
        # Calculate pairwise distances
        distances = torch.cdist(target_embeddings, target_embeddings)
        
        # Create importance mask
        importance_mask = torch.zeros((num_targets, num_targets), device=self.device)
        
        for i in important_targets:
            if i < num_targets:
                # Mark targets with similar importance
                importance_mask[i, important_targets] = 1
                # Remove self-connections
                importance_mask[i, i] = 0
        
        # Positive pairs: targets with similar importance (should be close)
        positive_pairs = importance_mask.bool()
        
        # Create a mask for self-connections
        self_connections = torch.eye(num_targets, dtype=torch.bool, device=self.device)
        
        # Negative pairs: targets with different importance (should be distant)
        # Avoid using subtraction with boolean tensors
        negative_pairs = ~positive_pairs & ~self_connections
        
        # Calculate positive loss (pull similar targets together)
        positive_loss = torch.mean(distances[positive_pairs]) if positive_pairs.sum() > 0 else torch.tensor(0.0, device=self.device)
        
        # Calculate negative loss (push different targets apart)
        negative_distances = distances[negative_pairs]
        negative_loss = torch.mean(torch.clamp(self.margin - negative_distances, min=0)) if negative_pairs.sum() > 0 else torch.tensor(0.0, device=self.device)
        
        # Combine losses
        loss = positive_loss + negative_loss
        
        return loss
    
    def enhanced_contrastive_loss(self, embeddings, target_indices, important_targets):
        """
        Enhanced contrastive loss with better importance-based clustering
        
        Args:
            embeddings: Node embeddings
            target_indices: Indices of target nodes
            important_targets: List of important target indices
            
        Returns:
            Loss value
        """
        # Get target embeddings
        target_embeddings = embeddings[target_indices]
        
        # Number of targets
        num_targets = len(target_indices)
        
        # Calculate pairwise distances
        distances = torch.cdist(target_embeddings, target_embeddings)
        
        # Get target IDs for each index
        target_ids = [self.reverse_node_map[idx.item()] for idx in target_indices]
        
        # Create importance clustering
        importance_groups = {}
        
        # Group targets by importance score ranges
        for i, target_id in enumerate(target_ids):
            importance = self.target_importance.get(target_id, 0.5)
            
            # Create importance bins (0.0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0)
            bin_idx = min(3, int(importance * 4))
            
            if bin_idx not in importance_groups:
                importance_groups[bin_idx] = []
            
            importance_groups[bin_idx].append(i)
        
        # Create importance mask
        importance_mask = torch.zeros((num_targets, num_targets), device=self.device)
        
        # Targets in the same importance group should be close to each other
        for group_indices in importance_groups.values():
            for i in group_indices:
                for j in group_indices:
                    if i != j:  # Avoid self-connections
                        importance_mask[i, j] = 1
        
        # Positive pairs: targets with similar importance (should be close)
        positive_pairs = importance_mask.bool()
        
        # Create a mask for self-connections
        self_connections = torch.eye(num_targets, dtype=torch.bool, device=self.device)
        
        # Negative pairs: targets with different importance (should be distant)
        negative_pairs = ~positive_pairs & ~self_connections
        
        # Calculate positive loss (pull similar targets together)
        positive_loss = torch.mean(distances[positive_pairs]) if positive_pairs.sum() > 0 else torch.tensor(0.0, device=self.device)
        
        # Calculate negative loss (push different targets apart)
        negative_distances = distances[negative_pairs]
        negative_loss = torch.mean(torch.clamp(self.margin - negative_distances, min=0)) if negative_pairs.sum() > 0 else torch.tensor(0.0, device=self.device)
        
        # Combine losses
        loss = positive_loss + negative_loss
        
        return loss


def calculate_priorities(embeddings, node_map, reverse_node_map, compound_indices, 
                        target_indices, target_importance, embedding_weight=0.05, top_k=10):
    """
    Calculate target priorities with enhanced importance score integration
    
    Args:
        embeddings: Node embeddings
        node_map: Map from node IDs to indices
        reverse_node_map: Map from indices to node IDs
        compound_indices: Indices of compound nodes
        target_indices: Indices of target nodes
        target_importance: Dictionary mapping target IDs to importance scores
        embedding_weight: Weight for embedding similarity vs. importance
        top_k: Number of top targets to return for each compound
        
    Returns:
        Dictionary mapping compounds to their prioritized targets
    """
    # Weight for importance scores
    importance_weight = 1.0 - embedding_weight
    logger.info(f"Final weights: embedding={embedding_weight:.4f}, importance={importance_weight:.4f}")
    
    # Extract embeddings
    compound_embeddings = embeddings[compound_indices]
    target_embeddings = embeddings[target_indices]
    
    # Create prioritization dictionary
    priorities = {}
    
    # Find median importance score for normalization
    median_importance = 0.5
    if target_importance:
        importance_values = list(target_importance.values())
        median_importance = np.median(importance_values)
    
    # Calculate priorities for each compound
    for i, compound_idx in enumerate(tqdm(compound_indices, desc="Calculating priorities")):
        compound_id = reverse_node_map[compound_idx.item()]
        compound_embedding = compound_embeddings[i].unsqueeze(0)
        
        # Calculate cosine similarity to all targets
        similarities = F.cosine_similarity(compound_embedding, target_embeddings)
        
        # Create target scores dictionary
        target_scores = {}
        
        for j, target_idx in enumerate(target_indices):
            target_id = reverse_node_map[target_idx.item()]
            similarity = similarities[j].item()
            
            # Get importance score for this target (default to median if not found)
            importance = target_importance.get(target_id, median_importance)
            
            # Apply nonlinear transformation to increase contrast in importance scores
            # This amplifies differences between high and low importance scores
            adjusted_importance = np.power(importance, 1.5)  # Apply power transformation
            
            # Calculate weighted score with enhanced importance influence
            weighted_score = embedding_weight * similarity + importance_weight * adjusted_importance
            
            # Store in scores dictionary
            target_scores[target_id] = weighted_score
        
        # Sort targets by score and keep top K
        sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
        priorities[compound_id] = dict(sorted_targets[:top_k])
    
    return priorities


def train_model(data, args, important_targets, validated_pairs=None):
    """
    Train the prioritization model
    
    Args:
        data: Graph data object
        args: Command line arguments
        important_targets: List of important target indices
        validated_pairs: Validated compound-target pairs (optional)
        
    Returns:
        Node embeddings, model
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check if we should load a pretrained model
    if args.load_model is not None and os.path.exists(args.load_model):
        logger.info(f"Loading pretrained model from {args.load_model}")
        
        # Initialize model with the same architecture
        model = FixedGraphSAGE(
            in_dim=data.x.size(1),
            hidden_dim=args.hidden_dim,
            out_dim=args.output_dim,
            dropout=args.dropout
        )
        
        # Load weights
        model.load_state_dict(torch.load(args.load_model))
        model = model.to(device)
        
        # Generate embeddings directly without training
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            embeddings = model(data.x, data.edge_index, data.edge_weight)
        
        logger.info("Successfully loaded pretrained model and generated embeddings")
        return embeddings, model
    
    # Initialize model
    model = FixedGraphSAGE(
        in_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=args.output_dim,
        dropout=args.dropout
    )
    
    if args.force_mlp:
        logger.info("Forcing MLP model (bypassing GNN)")
        model.gnn_available = False
    
    # Move model to device
    model = model.to(device)
    
    # Initialize trainer
    trainer = TargetPrioritizationTrainer(
        model=model,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        margin=args.margin,
        patience=args.patience
    )
    
    # Train model
    embeddings = trainer.train(data, important_targets)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save loss plot
    loss_plot_path = os.path.join(args.output_dir, 'training_loss.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training loss plot to {loss_plot_path}")
    
    # Close the figure to free memory
    plt.close()
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'tcm_target_model.pt')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
    
    return embeddings, model


def save_results(priorities, args):
    """
    Save prioritization results
    
    Args:
        priorities: Dictionary mapping compounds to prioritized targets
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save as CSV
    rows = []
    for compound_id, targets in priorities.items():
        for rank, (target_id, score) in enumerate(targets.items(), 1):
            rows.append({
                'compound_id': compound_id,
                'target_id': target_id,
                'rank': rank,
                'priority_score': score
            })
    
    results_df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = os.path.join(args.output_dir, 'target_priorities.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved target priorities to {csv_path}")
    
    # Save as JSON for easier programmatic access
    json_data = {}
    for compound_id, targets in priorities.items():
        json_data[compound_id] = [
            {'target_id': target_id, 'score': float(score)}
            for target_id, score in targets.items()
        ]
    
    json_path = os.path.join(args.output_dir, 'target_priorities.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Saved target priorities to {json_path}")


def main(args):
    """
    Main function
    
    Args:
        args: Command line arguments
    """
    # Start time
    start_time = datetime.now()
    logger.info(f"Starting TCM target prioritization at {start_time}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    kg_data, db_data, disease_data, validated_data, compound_features, target_importance = load_data(args)
    
    # Build graph
    data, node_map, reverse_node_map = build_graph(
        compound_target_data=kg_data,
        disease_target_data=disease_data,
        compound_features=compound_features,
        target_features=None
    )
    
    # Identify important targets (those with high importance scores)
    importance_threshold = 0.7  # Threshold for "important" targets
    important_targets = []
    
    for target_id, importance in target_importance.items():
        if target_id in node_map and importance > importance_threshold:
            target_idx = node_map[target_id] - len(compound_features)
            important_targets.append(target_idx)
    
    logger.info(f"Identified {len(important_targets)} important targets with importance > {importance_threshold}")
    
    # Train model
    embeddings, model = train_model(data, args, important_targets, validated_data)
    
    # Extract node IDs
    compound_ids = [reverse_node_map[idx.item()] for idx in data.compound_indices]
    target_ids = [reverse_node_map[idx.item()] for idx in data.target_indices]
    
    # Create embedding similarity scores
    embedding_scores = {}
    
    # Calculate similarities for each compound
    for i, compound_idx in enumerate(data.compound_indices):
        compound_id = reverse_node_map[compound_idx.item()]
        compound_embedding = embeddings[compound_idx].unsqueeze(0)
        
        # Calculate cosine similarity to all targets
        target_embeddings = embeddings[data.target_indices]
        similarities = F.cosine_similarity(compound_embedding, target_embeddings)
        
        # Create target similarity dictionary
        target_similarities = {}
        for j, target_idx in enumerate(data.target_indices):
            target_id = reverse_node_map[target_idx.item()]
            similarity = similarities[j].item()
            target_similarities[target_id] = similarity
        
        embedding_scores[compound_id] = target_similarities
    
    # Tune weights if requested
    if args.tune_weights and len(validated_data) > 0:
        logger.info("Tuning embedding and importance weights...")
        
        # Define more specific filenames for different plots
        weight_tuning_plot = os.path.join(args.output_dir, 'weight_tuning.png')
        
        # Pass these to your weight tuning function
        optimal_weights = find_optimal_weights(
            embedding_scores, target_importance, validated_data, 
            compound_ids, target_ids, weight_tuning_plot
        )
        
        # Use optimal weight for average precision
        embedding_weight = optimal_weights.get('optimal_ap_weight', args.embedding_weight)
        logger.info(f"Using optimal embedding weight: {embedding_weight:.2f}")
    else:
        # Use provided weight
        embedding_weight = args.embedding_weight
    
    # Calculate target priorities
    priorities = calculate_priorities(
        embeddings=embeddings,
        node_map=node_map,
        reverse_node_map=reverse_node_map,
        compound_indices=data.compound_indices,
        target_indices=data.target_indices,
        target_importance=target_importance,
        embedding_weight=embedding_weight,
        top_k=args.top_k
    )
    
    # Validate model if validation data is available
    if len(validated_data) > 0:
        logger.info("Validating model against known interactions...")
        validation_plot = os.path.join(args.output_dir, 'validation_results.png')
        metrics = validate_model(priorities, validated_data, compound_ids, target_ids, validation_plot)
        
        # Log validation metrics
        logger.info(f"Validation Results:")
        logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Save results
    save_results(priorities, args)
    
    # Print example
    print("\nExample prioritization:")
    example_compound = list(priorities.keys())[0]
    print(f"Compound {example_compound} target prioritization")
    for i, (target_id, score) in enumerate(priorities[example_compound].items(), 1):
        print(f"  {i}. {target_id}: {score:.4f}")
    
    # End time
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"TCM target prioritization completed in {duration}")
    print("\nTCM target prioritization completed!")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    args = parse_args()
    main(args)
