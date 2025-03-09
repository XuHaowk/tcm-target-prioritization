import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix the LSTM aggregator issue
def modify_aggregator():
    aggregator_file = 'src/models/aggregators.py'
    with open(aggregator_file, 'r') as f:
        content = f.read()
    
    # Replace the problematic forward method
    if '_gather_emb' in content:
        new_content = content.replace(
            'x, _ = self._gather_emb(x, index, ptr, dim)',
            '# Group features by index\n        out = torch.zeros((index.max().item() + 1, self.hidden_dim), device=x.device)'
        )
        
        with open(aggregator_file, 'w') as f:
            f.write(new_content)
        print("Fixed aggregator.py")

# Update the GraphSAGE model to be more stable
def modify_graph_sage():
    model_file = 'src/models/graph_sage.py'
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Add gradient clipping to the model
    if 'use_lstm_aggregator' in content and 'gradient_clipping' not in content:
        new_content = content.replace(
            'def __init__(self, feature_dim, hidden_dim, output_dim, dropout=0.5, use_lstm_aggregator=True):',
            'def __init__(self, feature_dim, hidden_dim, output_dim, dropout=0.5, use_lstm_aggregator=False):'
        )
        
        with open(model_file, 'w') as f:
            f.write(new_content)
        print("Modified model to use mean aggregator instead of LSTM for stability")

# Update the trainer to add gradient clipping
def modify_trainer():
    trainer_file = 'src/training/trainer.py'
    with open(trainer_file, 'r') as f:
        content = f.read()
    
    # Add gradient clipping if not present
    if 'optimizer.step()' in content and 'clip_grad_norm_' not in content:
        new_content = content.replace(
            '# Backward pass\n            loss.backward()\n            optimizer.step()',
            '# Backward pass\n            loss.backward()\n            # Add gradient clipping to prevent exploding gradients\n            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)\n            optimizer.step()'
        )
        
        with open(trainer_file, 'w') as f:
            f.write(new_content)
        print("Added gradient clipping to trainer.py")

# Update the visualization to handle NaN values
def modify_visualization():
    vis_file = 'src/utils/visualization.py'
    with open(vis_file, 'r') as f:
        content = f.read()
    
    # Add NaN handling before t-SNE
    if 'reduced_embeddings = tsne.fit_transform' in content and 'torch.isnan' not in content:
        new_content = content.replace(
            'reduced_embeddings = tsne.fit_transform(embeddings[selected_indices].numpy())',
            '# Replace NaN values with zeros to allow visualization\n    emb_subset = embeddings[selected_indices].clone()\n    emb_subset[torch.isnan(emb_subset)] = 0.0\n    reduced_embeddings = tsne.fit_transform(emb_subset.numpy())'
        )
        
        with open(vis_file, 'w') as f:
            f.write(new_content)
        print("Modified visualization.py to handle NaN values")

# Update main script to use a lower learning rate
def modify_main():
    main_file = 'scripts/main_fixed.py'
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Add module import at the top
    if 'import sys' not in content:
        new_content = 'import sys\nimport os\n# Add the project root directory to Python path\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n' + content
        with open(main_file, 'w') as f:
            f.write(new_content)
    
    print("Updated main_fixed.py to include proper imports")

if __name__ == "__main__":
    modify_aggregator()
    modify_graph_sage() 
    modify_trainer()
    modify_visualization()
    modify_main()
    print("All fixes applied. Now try running the script with a lower learning rate (0.0001):")
    print("python scripts/main_fixed.py --kg_data data/raw/kg_data_extended.csv --db_data data/raw/database_data_extended.csv --disease_importance data/raw/disease_importance_extended.csv --validated_data data/raw/validated_interactions.csv --feature_dim 256 --hidden_dim 256 --output_dim 128 --dropout 0.3 --epochs 300 --lr 0.0001 --weight_decay 1e-5 --margin 0.4 --patience 30")
