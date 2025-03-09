import pandas as pd
import numpy as np
import torch
import os
import json
import argparse
from sklearn.model_selection import KFold
from tqdm import tqdm
import subprocess
import sys

def run_cross_validation(args):
    """Run k-fold cross-validation"""
    print(f"Running {args.folds}-fold cross-validation...")
    
    # Load validated target data
    validated_file = 'data/processed/validated_reference.csv'
    if not os.path.exists(validated_file):
        print("Validated reference data not found. Please run integrate_validated_data.py first.")
        return
    
    validated_df = pd.read_csv(validated_file)
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    cv_results = []
    
    # Generate cross-validation datasets
    compound_target_pairs = list(zip(validated_df['compound'], validated_df['target']))
    fold_idx = 1
    
    for train_idx, test_idx in kf.split(compound_target_pairs):
        print(f"\n===== Running Fold {fold_idx} =====")
        
        # Prepare train and test sets
        train_pairs = [compound_target_pairs[i] for i in train_idx]
        test_pairs = [compound_target_pairs[i] for i in test_idx]
        
        # Create fold directory
        fold_dir = f"data/cv_fold_{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save train and test sets
        train_df = pd.DataFrame(train_pairs, columns=['compound', 'target'])
        test_df = pd.DataFrame(test_pairs, columns=['compound', 'target'])
        
        # Add confidence scores
        train_df = train_df.merge(validated_df[['compound', 'target', 'confidence_score']], 
                                 on=['compound', 'target'], how='left')
        test_df = test_df.merge(validated_df[['compound', 'target', 'confidence_score']], 
                               on=['compound', 'target'], how='left')
        
        train_df.to_csv(f"{fold_dir}/cv_train.csv", index=False)
        test_df.to_csv(f"{fold_dir}/cv_test.csv", index=False)
        
        # Run training with parameters
        train_cmd = [
            "python", "scripts/main.py",
            "--kg_data", "data/raw/kg_data_extended.csv",
            "--db_data", "data/raw/database_data_extended.csv",
            "--disease_importance", "data/raw/disease_importance_extended.csv",
            "--feature_dim", str(args.feature_dim),
            "--hidden_dim", str(args.hidden_dim),
            "--output_dim", str(args.output_dim),
            "--dropout", str(args.dropout),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--margin", str(args.margin),
            "--patience", str(args.patience),
            "--cv_fold", str(fold_idx),
            "--cv_train_file", f"{fold_dir}/cv_train.csv",
            "--cv_test_file", f"{fold_dir}/cv_test.csv"
        ]
        
        if args.use_lstm_aggregator:
            train_cmd.append("--use_lstm_aggregator")
        
        print(f"Running training command: {' '.join(train_cmd)}")
        subprocess.run(train_cmd)
        
        # Load and evaluate current fold results
        metrics_file = f"results/cv_fold_{fold_idx}/metrics.csv"
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            metrics_df['Fold'] = fold_idx
            cv_results.append(metrics_df)
        else:
            print(f"Warning: Could not find metrics file for fold {fold_idx}.")
        
        fold_idx += 1
    
    # Combine all fold results
    if cv_results:
        all_cv_results = pd.concat(cv_results)
        
        # Calculate mean and standard deviation
        mean_results = all_cv_results.drop(columns=['Fold']).mean()
        std_results = all_cv_results.drop(columns=['Fold']).std()
        
        # Create final results dataframe
        final_results = pd.DataFrame({
            'Metric': mean_results.index,
            'Mean': mean_results.values,
            'Std': std_results.values
        })
        
        # Save cross-validation results
        os.makedirs('results/cross_validation', exist_ok=True)
        all_cv_results.to_csv('results/cross_validation/cv_metrics.csv', index=False)
        final_results.to_csv('results/cross_validation/cv_summary.csv', index=False)
        
        # Print summary results
        print("\n===== Cross-Validation Results Summary =====")
        for i, row in final_results.iterrows():
            print(f"{row['Metric']}: {row['Mean']:.4f} ± {row['Std']:.4f}")
    else:
        print("No cross-validation results available.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCM Target Prioritization Cross-Validation")
    
    # Cross-validation parameters
    parser.add_argument('--folds', type=int, default=5,
                      help='Number of cross-validation folds (default: 5)')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=256,
                      help='Feature dimension (default: 256)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden layer dimension (default: 256)')
    parser.add_argument('--output_dim', type=int, default=128,
                      help='Output embedding dimension (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout probability (default: 0.3)')
    parser.add_argument('--use_lstm_aggregator', action='store_true',
                      help='Use LSTM aggregator (default: False)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300,
                      help='Number of training epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay (default: 1e-5)')
    parser.add_argument('--margin', type=float, default=0.4,
                      help='Contrastive loss margin (default: 0.4)')
    parser.add_argument('--patience', type=int, default=30,
                      help='Early stopping patience (default: 30)')
    
    args = parser.parse_args()
    run_cross_validation(args)
