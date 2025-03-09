#!/usr/bin/env python3
"""
Enhanced TCM Target Prioritization Main Script
"""

import os
import sys
# Add the project root directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# SIMPLIFIED DATA LOADING FUNCTIONS
# These will be used if imports fail

def load_knowledge_graph(file_path):
    """Load knowledge graph data from CSV file"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return pd.DataFrame()
    
    try:
        kg_data = pd.read_csv(file_path)
        print(f"Successfully loaded data: {file_path}")
        return kg_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_database(file_path):
    """Load compound database data from CSV file"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return pd.DataFrame()
    
    try:
        db_data = pd.read_csv(file_path)
        print(f"Successfully loaded data: {file_path}")
        return db_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_disease_importance(file_path):
    """Load disease importance data from CSV file"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return pd.DataFrame()
    
    try:
        importance_data = pd.read_csv(file_path)
        print(f"Successfully loaded data: {file_path}")
        return importance_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_validated_interactions(file_path):
    """Load validated compound-target interactions"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return []
    
    try:
        validated_data = pd.read_csv(file_path)
        validated_pairs = list(zip(validated_data['compound_id'], validated_data['target_id']))
        print(f"Successfully loaded data: {file_path}")
        return validated_pairs
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

# Try to import modules, fall back to simplified versions if they fail
try:
    from src.data.kg_processor import load_knowledge_graph
    from src.data.database_processor import load_database
    from src.data.disease_importance_processor import load_disease_importance 
    from src.data.validated_data_processor import load_validated_interactions
    print("Successfully imported data processing modules")
except ImportError as e:
    print(f"Warning: Using simplified data loading functions: {e}")
    # We're already using the fallback functions defined above

# Continue with the rest of your imports and code...
