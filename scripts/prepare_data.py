import os
import pandas as pd
import argparse
import random
import numpy as np

def check_and_create_directories():
    """Check and create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'results/models',
        'results/embeddings',
        'results/prioritization',
        'results/evaluation',
        'results/visualizations'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def check_required_files():
    """Check if required files exist"""
    required_files = [
        'data/raw/kg_data_extended.csv',
        'data/raw/database_data_extended.csv',
        'data/raw/disease_importance_extended.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Warning: The following required files do not exist:")
        for file_path in missing_files:
            print(f" - {file_path}")
        print("Please create these files before running the main program.")
        return False
    
    return True

def generate_sample_data():
    """Generate sample data files if needed"""
    # 1. Generate knowledge graph data
    if not os.path.exists('data/raw/kg_data_extended.csv'):
        # List of compounds
        compounds = [
            "Berberine", "Curcumin", "Ginsenoside_Rg1", "Astragaloside_IV", 
            "Baicalein", "Quercetin", "Tanshinone_IIA", "Tetrandrine", 
            "Emodin", "Resveratrol", "Huperzine_A", "Baicalin", 
            "Ginsenoside_Rb1", "Ligustrazine", "Loganin"
        ]
        
        # List of targets
        targets = []
        # Inflammation targets
        targets.extend(["TNF", "IL1B", "IL6", "NFKB1", "PTGS2", "IL10", "CXCL8", "CCL2", "IL17A", "IL4", "IL13"])
        # Apoptosis/survival targets
        targets.extend(["CASP3", "BCL2", "BAX", "TP53", "AKT1", "MAPK1", "JUN", "STAT3", "JAK2"])
        # Antioxidant targets
        targets.extend(["SOD1", "CAT", "GPX1", "NRF2", "HMOX1"])
        # Fibrosis targets
        targets.extend(["TGFB1", "MMP9", "COL1A1", "SMAD3", "SERPINE1", "CTGF", "TIMP1", "ACTA2", "FN1"])
        # Vascular targets
        targets.extend(["VEGFA", "NOS3", "HIF1A", "ANGPT1", "KDR"])
        # Metabolic targets
        targets.extend(["PPARG", "AMPK", "LDLR", "SREBF1", "GLUT4", "INSR", "ADIPOQ"])
        # Neural targets
        targets.extend(["BDNF", "NGF", "ACHE", "GSK3B", "APP", "MAPT", "CHAT"])
        
        # Relation types
        relation_types = ['activation', 'inhibition', 'binding', 'regulation']
        
        # Generate random relations
        kg_data = []
        for compound in compounds:
            # Number of targets per compound varies
            num_targets = random.randint(5, 15)
            selected_targets = random.sample(targets, num_targets)
            
            for target in selected_targets:
                relation_type = random.choice(relation_types)
                relation_strength = round(random.uniform(0.5, 0.95), 2)
                
                kg_data.append({
                    'compound': compound,
                    'target': target,
                    'relation_type': relation_type,
                    'relation_strength': relation_strength
                })
        
        kg_df = pd.DataFrame(kg_data)
        kg_df.to_csv('data/raw/kg_data_extended.csv', index=False)
        print("Generated knowledge graph sample data: data/raw/kg_data_extended.csv")
    
    # 2. Generate database data
    if not os.path.exists('data/raw/database_data_extended.csv'):
        # Use compounds and targets from above
        # Create some new combinations
        db_data = []
        for compound in compounds:
            # Number of targets varies
            num_targets = random.randint(8, 20)
            selected_targets = random.sample(targets, num_targets)
            
            for target in selected_targets:
                confidence_score = round(random.uniform(0.55, 0.95), 2)
                
                db_data.append({
                    'compound': compound,
                    'target': target,
                    'confidence_score': confidence_score
                })
        
        db_df = pd.DataFrame(db_data)
        db_df.to_csv('data/raw/database_data_extended.csv', index=False)
        print("Generated database sample data: data/raw/database_data_extended.csv")
    
    # 3. Generate disease importance data
    if not os.path.exists('data/raw/disease_importance_extended.csv'):
        # List of diseases
        diseases = ["Silicosis", "Liver_Fibrosis", "IBD", "Diabetes", "Alzheimers", "Rheumatoid_Arthritis"]
        
        # Generate disease-specific target importance
        importance_data = []
        
        for disease in diseases:
            # Select subset of targets relevant to this disease
            num_targets = random.randint(20, 35)
            selected_targets = random.sample(targets, num_targets)
            
            # Assign importance scores with some being more important
            for i, target in enumerate(selected_targets):
                # First few targets get higher scores
                if i < 5:
                    importance_score = round(random.uniform(0.85, 0.97), 2)
                elif i < 10:
                    importance_score = round(random.uniform(0.75, 0.85), 2)
                elif i < 20:
                    importance_score = round(random.uniform(0.65, 0.75), 2)
                else:
                    importance_score = round(random.uniform(0.5, 0.65), 2)
                
                importance_data.append({
                    'disease': disease,
                    'target': target,
                    'importance_score': importance_score
                })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df.to_csv('data/raw/disease_importance_extended.csv', index=False)
        print("Generated disease importance sample data: data/raw/disease_importance_extended.csv")
    
    # 4. Generate validated interactions data
    if not os.path.exists('data/raw/validated_interactions.csv'):
        # Generate a subset of validated interactions
        validated_data = []
        
        # Select some compound-target pairs as validated
        for compound in compounds:
            # Number of validated targets per compound
            num_targets = random.randint(3, 7)
            selected_targets = random.sample(targets, num_targets)
            
            for target in selected_targets:
                confidence_score = round(random.uniform(0.75, 0.98), 2)
                validation_methods = ['enzymatic_assay', 'western_blot', 'reporter_assay', 
                                      'ELISA', 'qRT-PCR', 'phosphorylation_assay']
                
                validated_data.append({
                    'compound': compound,
                    'target': target,
                    'validation_method': random.choice(validation_methods),
                    'confidence_score': confidence_score,
                    'reference_source': f"PMID:{random.randint(20000000, 30000000)}"
                })
        
        validated_df = pd.DataFrame(validated_data)
        validated_df.to_csv('data/raw/validated_interactions.csv', index=False)
        print("Generated validated interactions sample data: data/raw/validated_interactions.csv")

def main(args):
    """Main function"""
    print("Preparing data...")
    
    # Check and create directories
    check_and_create_directories()
    
    # Generate sample data if specified
    if args.generate_samples:
        generate_sample_data()
    
    # Check required files
    files_exist = check_required_files()
    
    if files_exist:
        print("Data preparation complete.")
    else:
        print("Please prepare the required data files before running the main program.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TCM target prioritization data")
    parser.add_argument('--generate_samples', action='store_true',
                      help='Generate sample data files')
    
    args = parser.parse_args()
    main(args)
