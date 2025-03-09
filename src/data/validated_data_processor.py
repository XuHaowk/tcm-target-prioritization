"""
Validated data processor module
"""

import pandas as pd
import os

def load_validated_interactions(file_path):
    """
    Load validated compound-target interactions
    
    Args:
        file_path: Path to validated interactions CSV file
        
    Returns:
        List of (compound_id, target_id) tuples
    """
    if not os.path.exists(file_path):
        print(f"Error: Validated data file {file_path} does not exist")
        return []
    
    try:
        validated_data = pd.read_csv(file_path)
        
        required_columns = ['compound_id', 'target_id']
        missing_columns = [col for col in required_columns if col not in validated_data.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns in validated data: {missing_columns}")
            return []
            
        validated_pairs = list(zip(validated_data['compound_id'], validated_data['target_id']))
        
        print(f"Loaded {len(validated_pairs)} validated compound-target interactions")
        return validated_pairs
    
    except Exception as e:
        print(f"Error loading validated interaction data: {e}")
        return []
