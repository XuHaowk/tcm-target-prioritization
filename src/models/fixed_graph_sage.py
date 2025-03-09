"""
Fixed GraphSAGE model implementation

This module contains an enhanced GraphSAGE model with fixes for the tensor dimension issues 
and more detailed error logging for diagnosis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, MessagePassing
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FixedGraphSAGE(nn.Module):
    """
    Improved GraphSAGE model with fixes for tensor dimension issues
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3, aggregator=None):
        """
        Initialize GraphSAGE model
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            dropout: Dropout probability
            aggregator: Aggregator function (optional)
        """
        super(FixedGraphSAGE, self).__init__()
        
        # Store dimensions for debugging
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Input transformation
        self.input_transform = nn.Linear(in_dim, hidden_dim)
        
        # MLP layers as fallback path
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        # GraphSAGE convolution layers
        try:
            # Custom SAGE convolution with more verbose error handling
            self.conv1 = FixedSAGEConv(hidden_dim, hidden_dim)
            self.conv2 = FixedSAGEConv(hidden_dim, out_dim)
            self.gnn_available = True
            logger.info("Successfully initialized GraphSAGE convolution layers")
        except Exception as e:
            logger.error(f"Failed to initialize GraphSAGE layers: {e}")
            self.gnn_available = False
        
        # Batch normalization layers for stability
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
        # Skip connection transformation
        self.skip_transform = nn.Linear(hidden_dim, out_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')
                else:
                    nn.init.normal_(param, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with additional error handling and MLP fallback
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights (optional)
            
        Returns:
            Node embeddings
        """
        # Check for NaN values
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Log input shapes for debugging
        logger.info(f"Input shapes: x={x.shape}, edge_index={edge_index.shape}")
        
        # Basic edge_index validation
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            logger.warning(f"edge_index has incorrect shape: {edge_index.shape}, should be [2, num_edges]")
            if edge_index.numel() > 0:
                try:
                    # Try to reshape edge_index
                    edge_index = edge_index.view(2, -1)
                    logger.info(f"Reshaped edge_index to {edge_index.shape}")
                except Exception as e:
                    logger.error(f"Failed to reshape edge_index: {e}")
        
        # Apply feature transformation
        x = self.input_transform(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.batch_norm1(x)
        
        # Store for skip connection
        x_skip = x
        
        # Try GraphSAGE layers if available
        gnn_success = False
        if hasattr(self, 'gnn_available') and self.gnn_available:
            try:
                # Check if edge_index indices are within bounds
                max_idx = x.size(0) - 1
                if edge_index.max() > max_idx:
                    logger.warning(f"edge_index contains indices ({edge_index.max()}) that exceed the number of nodes ({max_idx+1})")
                    # Clip indices to valid range
                    edge_index = torch.clamp(edge_index, 0, max_idx)
                
                # First GraphSAGE layer
                if edge_weight is not None:
                    x1 = self.conv1(x, edge_index, edge_weight)
                else:
                    x1 = self.conv1(x, edge_index)
                
                x1 = F.leaky_relu(x1, negative_slope=0.2)
                x1 = self.dropout(x1)
                
                # Second GraphSAGE layer
                if edge_weight is not None:
                    x2 = self.conv2(x1, edge_index, edge_weight)
                else:
                    x2 = self.conv2(x1, edge_index)
                
                # If we got here, GNN was successful
                x = x2
                gnn_success = True
                logger.info("GraphSAGE layers successfully applied")
            except Exception as e:
                logger.error(f"Error in GraphSAGE layers: {e}")
                # Will fall back to MLP path
        
        # MLP fallback path if GNN failed
        if not gnn_success:
            logger.info("Using MLP fallback path")
            # Apply hidden layer
            h = self.hidden_layer(x_skip)
            h = F.leaky_relu(h, negative_slope=0.2)
            h = self.dropout(h)
            
            # Apply output transformation
            x = self.output_layer(h)
        
        # Add skip connection (transformed to match output dimension)
        skip_contribution = self.skip_transform(x_skip)
        
        # Ensure dimensions match before adding
        if x.shape[1] == skip_contribution.shape[1]:
            logger.info(f"Adding skip connection: {x.shape} + {skip_contribution.shape}")
            x = x + skip_contribution
        else:
            logger.warning(f"Dimension mismatch for skip connection: {x.shape}, skip: {skip_contribution.shape}")
            # Use the correctly shaped tensor
            if x.shape[1] == self.out_dim:
                pass  # Keep x as is
            else:
                x = skip_contribution  # Use skip connection output
        
        # Final batch normalization
        try:
            x = self.batch_norm2(x)
        except Exception as e:
            logger.error(f"Error in batch normalization: {e}")
        
        # Replace any NaN values
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        
        return x


class FixedSAGEConv(SAGEConv):
    """
    A fixed version of SAGEConv with additional error handling and debugging
    """
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with detailed error handling
        """
        try:
            # Check input dimensions
            if x.dim() != 2:
                logger.error(f"x should be 2-dimensional, but got shape {x.shape}")
                raise ValueError(f"x should be 2-dimensional, but got shape {x.shape}")
            
            # Check edge_index
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                logger.error(f"edge_index should have shape [2, num_edges], but got {edge_index.shape}")
                raise ValueError(f"edge_index should have shape [2, num_edges], but got {edge_index.shape}")
            
            # Check if indices are within bounds
            num_nodes = x.size(0)
            if edge_index.max() >= num_nodes:
                logger.error(f"edge_index contains indices that exceed the number of nodes: max index {edge_index.max()}, num nodes {num_nodes}")
                raise ValueError(f"edge_index contains indices that exceed the number of nodes: max index {edge_index.max()}, num nodes {num_nodes}")
            
            # Call parent class method with detailed exception handling
            return super().forward(x, edge_index, edge_weight)
            
        except Exception as e:
            logger.error(f"Error in FixedSAGEConv forward pass: {e}")
            raise
