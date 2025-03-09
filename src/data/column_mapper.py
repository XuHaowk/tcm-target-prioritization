# Create this as src/data/column_mapper.py
"""
Column mapping utilities to handle different data formats
"""

# Standard column names expected by the model
STANDARD_COLUMNS = {
    'compound_id': ['compound_id', 'compound', 'comp_id', 'molecule_id'],
    'target_id': ['target_id', 'target', 'protein_id', 'gene_id'],
    'disease_id': ['disease_id', 'disease', 'disorder_id', 'condition'],
    'relation_type': ['relation_type', 'interaction_type', 'type'],
    'importance_score': ['importance_score', 'importance', 'score', 'weight']
}

def find_matching_column(df, standard_name):
    """
    Find a column in the DataFrame that matches the standard name
    
    Args:
        df: DataFrame to search
        standard_name: Standard column name to look for
        
    Returns:
        Matching column name or None if not found
    """
    # First, check if the standard name itself exists
    if standard_name in df.columns:
        return standard_name
    
    # Check alternative names
    alternatives = STANDARD_COLUMNS.get(standard_name, [])
    for alt in alternatives:
        if alt in df.columns:
            print(f"Using column '{alt}' for '{standard_name}'")
            return alt
    
    return None

def extract_data_with_mapping(df, column_mapping):
    """
    Extract data from DataFrame using column mapping
    
    Args:
        df: DataFrame to extract from
        column_mapping: Dictionary mapping standard names to actual names
    
    Returns:
        DataFrame with renamed columns
    """
    result = df.copy()
    
    # Rename columns according to mapping
    for standard_name, actual_name in column_mapping.items():
        if actual_name is not None and actual_name in df.columns:
            result.rename(columns={actual_name: standard_name}, inplace=True)
    
    return result
