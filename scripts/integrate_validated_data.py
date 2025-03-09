import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def integrate_validated_data():
    """Integrate validated drug-target data and prepare train/validation/test sets"""
    
    # Check if files exist
    validated_file = 'data/raw/validated_interactions.csv'
    split_file = 'data/raw/data_split.csv'
    
    if not os.path.exists(validated_file):
        print(f"Validated interactions file not found: {validated_file}")
        print("Please create it first.")
        return
    
    # Load validated interactions data
    validated_df = pd.read_csv(validated_file)
    
    # Create processed data directory
    os.makedirs('data/processed', exist_ok=True)
    
    # If split file exists, use predefined splits
    if os.path.exists(split_file):
        split_df = pd.read_csv(split_file)
        
        # Create train/validation/test sets based on predefined splits
        train_pairs = []
        validation_pairs = []
        test_pairs = []
        
        for _, row in split_df.iterrows():
            compound = row['compound']
            target = row['target']
            split_type = row['split_type']
            
            if split_type == 'training':
                train_pairs.append((compound, target))
            elif split_type == 'validation':
                validation_pairs.append((compound, target))
            elif split_type == 'testing':
                test_pairs.append((compound, target))
        
        # Create data frames
        train_data = []
        for compound, target in train_pairs:
            # Get confidence score from validated data
            confidence = validated_df[(validated_df['compound'] == compound) & 
                                     (validated_df['target'] == target)]['confidence_score'].values
            
            if len(confidence) > 0:
                train_data.append({
                    'compound': compound,
                    'target': target,
                    'confidence_score': confidence[0]
                })
        
        validation_data = []
        for compound, target in validation_pairs:
            confidence = validated_df[(validated_df['compound'] == compound) & 
                                     (validated_df['target'] == target)]['confidence_score'].values
            
            if len(confidence) > 0:
                validation_data.append({
                    'compound': compound,
                    'target': target,
                    'confidence_score': confidence[0]
                })
        
        test_data = []
        for compound, target in test_pairs:
            confidence = validated_df[(validated_df['compound'] == compound) & 
                                     (validated_df['target'] == target)]['confidence_score'].values
            
            if len(confidence) > 0:
                test_data.append({
                    'compound': compound,
                    'target': target,
                    'confidence_score': confidence[0]
                })
    else:
        # No split file, create random train/validation/test split
        print("No predefined split file found. Creating random split.")
        
        # Extract compound-target pairs
        pairs = validated_df[['compound', 'target', 'confidence_score']]
        
        # Split into training (70%), validation (15%), and test (15%) sets
        train_pairs, temp_pairs = train_test_split(pairs, test_size=0.3, random_state=42)
        validation_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)
        
        train_data = train_pairs.to_dict('records')
        validation_data = validation_pairs.to_dict('records')
        test_data = test_pairs.to_dict('records')
    
    # Convert to data frames
    train_df = pd.DataFrame(train_data)
    validation_df = pd.DataFrame(validation_data)
    test_df = pd.DataFrame(test_data)
    
    # Save to CSV
    train_df.to_csv('data/processed/validated_train.csv', index=False)
    validation_df.to_csv('data/processed/validated_validation.csv', index=False)
    test_df.to_csv('data/processed/validated_test.csv', index=False)
    
    print(f"Training set size: {len(train_df)} pairs")
    print(f"Validation set size: {len(validation_df)} pairs")
    print(f"Test set size: {len(test_df)} pairs")
    
    # Create a merged reference data frame for performance evaluation
    reference_data = pd.concat([validation_df, test_df])
    reference_data.to_csv('data/processed/validated_reference.csv', index=False)
    
    print(f"Created merged reference data set: {len(reference_data)} pairs")
    print("Validated data integration complete!")

if __name__ == "__main__":
    integrate_validated_data()
