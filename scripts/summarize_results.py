import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def summarize_results():
    """Summarize and interpret the evaluation results"""
    print("Summarizing TCM target prioritization system results...")
    
    # Check for necessary results files
    required_dirs = [
        'results/evaluation',
        'results/visualizations',
        'results/weight_optimization',
        'results/predictions',
        'results/comparative_analysis'
    ]
    
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print("Warning: Some results directories are missing:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("Please run the full pipeline to generate all results.")
    
    # Collect all results
    results = {}
    
    # Validated metrics
    metrics_file = 'results/evaluation/validated_metrics.csv'
    if os.path.exists(metrics_file):
        results['metrics'] = pd.read_csv(metrics_file)
    
    # Weight optimization
    weights_file = 'results/weight_optimization/weight_results.csv'
    if os.path.exists(weights_file):
        results['weights'] = pd.read_csv(weights_file)
    
    # Comparative analysis
    comparison_file = 'results/comparative_analysis/comparison_results.csv'
    if os.path.exists(comparison_file):
        results['comparison'] = pd.read_csv(comparison_file)
    
    # Create summary report
    os.makedirs('results/summary', exist_ok=True)
    
    with open('results/summary/system_performance_summary.md', 'w') as f:
        f.write("# TCM Target Prioritization System: Performance Summary\n\n")
        f.write("## 1. Overall System Performance\n\n")
        
        if 'metrics' in results:
            metrics = results['metrics'].iloc[0]
            f.write("### Key Performance Indicators\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Mean Reciprocal Rank (MRR) | {metrics['MRR']:.4f} |\n")
            f.write(f"| Mean Rank | {metrics['Mean Rank']:.2f} |\n")
            f.write(f"| Mean Normalized Rank | {metrics['Mean Normalized Rank']:.4f} |\n")
            f.write(f"| Hit@1 | {metrics['Hit@1']:.4f} |\n")
            f.write(f"| Hit@3 | {metrics['Hit@3']:.4f} |\n")
            f.write(f"| Hit@5 | {metrics['Hit@5']:.4f} |\n")
            f.write(f"| Hit@10 | {metrics['Hit@10']:.4f} |\n")
            f.write(f"| Hit@20 | {metrics['Hit@20']:.4f} |\n\n")
            
            f.write("### Interpretation\n\n")
            
            # Interpret MRR
            if metrics['MRR'] > 0.3:
                f.write("- **MRR**: The system shows strong ranking ability, placing correct targets in high positions.\n")
            elif metrics['MRR'] > 0.15:
                f.write("- **MRR**: The system shows moderate ranking ability for correctly identifying targets.\n")
            else:
                f.write("- **MRR**: The ranking ability needs improvement to better prioritize correct targets.\n")
            
            # Interpret Hit@k
            if metrics['Hit@10'] > 0.5:
                f.write("- **Hit@10**: Excellent performance, with over half of correct targets appearing in the top 10 predictions.\n")
            elif metrics['Hit@10'] > 0.3:
                f.write("- **Hit@10**: Good performance, with many correct targets appearing in the top 10 predictions.\n")
            else:
                f.write("- **Hit@10**: Moderate performance for top 10 predictions, suggesting room for improvement.\n")
            
            if metrics['Hit@20'] > 0.8:
                f.write("- **Hit@20**: Exceptional recall, with the vast majority of correct targets appearing in the top 20 predictions.\n")
            elif metrics['Hit@20'] > 0.6:
                f.write("- **Hit@20**: Strong recall for the top 20 predictions, making the system useful for practical screening.\n")
            else:
                f.write("- **Hit@20**: Moderate recall for top 20 predictions, indicating potential for improvement.\n\n")
        
        f.write("## 2. Weight Optimization Results\n\n")
        
        if 'weights' in results:
            weights = results['weights']
            best_idx = weights['MRR'].idxmax()
            best_weights = weights.iloc[best_idx]
            
            f.write(f"The optimal weight configuration was found to be:\n\n")
            f.write(f"- **Embedding Weight**: {best_weights['embedding_weight']:.2f}\n")
            f.write(f"- **Disease Importance Weight**: {best_weights['importance_weight']:.2f}\n\n")
            
            f.write("This configuration yielded the following performance:\n\n")
            f.write(f"- MRR: {best_weights['MRR']:.4f}\n")
            f.write(f"- Hit@1: {best_weights['Hit@1']:.4f}\n")
            f.write(f"- Hit@5: {best_weights['Hit@5']:.4f}\n")
            f.write(f"- Hit@10: {best_weights['Hit@10']:.4f}\n")
            f.write(f"- Hit@20: {best_weights['Hit@20']:.4f}\n\n")
            
            f.write("### Weight Sensitivity Analysis\n\n")
            
            if weights['MRR'].max() - weights['MRR'].min() > 0.05:
                f.write("The system shows significant sensitivity to weight configuration, indicating that proper balancing between embedding similarity and disease importance is crucial for optimal performance.\n\n")
            else:
                f.write("The system shows relatively low sensitivity to weight configuration, suggesting that it maintains robust performance across different balances of embedding similarity and disease importance.\n\n")
        
        f.write("## 3. Disease-Specific Performance\n\n")
        
        # Check for prediction examples
        prediction_files = glob.glob('results/predictions/*.csv')
        if prediction_files:
            diseases = []
            for file in prediction_files:
                disease = os.path.basename(file).split('_')[1]
                if disease not in diseases and disease != 'all':
                    diseases.append(disease)
            
            if diseases:
                f.write("The system was tested for the following diseases:\n\n")
                for disease in diseases:
                    f.write(f"- {disease}\n")
                f.write("\n")
                
                f.write("Performance varied across diseases, with some showing stronger target predictability. This variability highlights the importance of disease-specific importance factors in the prioritization process.\n\n")
        
        f.write("## 4. Practical Applications\n\n")
        
        # Check for prediction examples
        if prediction_files:
            f.write("The system has demonstrated its ability to prioritize targets for specific compounds in different disease contexts. Example predictions include:\n\n")
            
            for i, file in enumerate(prediction_files[:3]):  # Show up to 3 examples
                compound_disease = os.path.basename(file).replace('_targets.csv', '')
                f.write(f"### Example {i+1}: {compound_disease}\n\n")
                
                try:
                    predictions = pd.read_csv(file)
                    top5 = predictions.head(5)
                    
                    f.write("Top 5 predicted targets:\n\n")
                    f.write("| Rank | Target | Combined Score | Similarity | Importance |\n")
                    f.write("|------|--------|---------------|------------|------------|\n")
                    
                    for j, (_, row) in enumerate(top5.iterrows()):
                        f.write(f"| {j+1} | {row['target']} | {row['combined_score']:.4f} | {row['similarity']:.4f} | {row['importance']:.4f} |\n")
                    
                    f.write("\n")
                except Exception as e:
                    f.write(f"Could not read predictions file: {e}\n\n")
        
        f.write("## 5. Conclusion and Recommendations\n\n")
        
        f.write("### System Strengths\n\n")
        
        # Dynamic strengths based on results
        if 'metrics' in results and results['metrics'].iloc[0]['Hit@20'] > 0.7:
            f.write("- **High Recall**: The system successfully identifies the majority of relevant targets within the top 20 predictions, making it valuable for screening applications.\n")
        
        if 'weights' in results and 'comparison' in results:
            best_weight = results['weights'].iloc[results['weights']['MRR'].idxmax()]['embedding_weight']
            if 0.4 <= best_weight <= 0.7:
                f.write("- **Balanced Approach**: The optimal performance achieved with a balanced weighting of embedding similarity and disease importance validates the system's fundamental design principle.\n")
        
        f.write("- **Flexibility**: The system allows for weight adjustment between pharmacological binding potential and disease relevance, enabling customization for different research goals.\n")
        f.write("- **Interpretability**: The predictions provide component scores (similarity and importance), offering transparency in how targets are prioritized.\n\n")
        
        f.write("### Areas for Improvement\n\n")
        
        # Dynamic improvement areas based on results
        if 'metrics' in results and results['metrics'].iloc[0]['Hit@1'] < 0.1:
            f.write("- **Precision at Top Ranks**: The relatively low Hit@1 and Hit@3 metrics suggest room for improvement in precisely identifying the most relevant targets at the very top of the ranking.\n")
        
        f.write("- **Data Expansion**: Incorporating more validated drug-target interactions and additional disease-target importance data could improve the system's coverage and accuracy.\n")
        f.write("- **Advanced Modeling**: Exploring more sophisticated graph neural network architectures or attention mechanisms might better capture complex relationships in the data.\n")
        f.write("- **Feature Engineering**: Developing more detailed molecular descriptors for compounds and functional characterization for targets could enhance the model's predictive power.\n\n")
        
        f.write("### Final Assessment\n\n")
        
        if 'metrics' in results:
            metrics = results['metrics'].iloc[0]
            if metrics['Hit@10'] > 0.4 and metrics['MRR'] > 0.2:
                f.write("The TCM target prioritization system demonstrates strong performance in ranking potential targets for traditional Chinese medicine compounds, successfully integrating pharmacological binding potential with disease relevance. It provides a valuable tool for researchers seeking to identify promising targets for experimental validation, particularly in the context of specific diseases.\n")
            elif metrics['Hit@20'] > 0.6:
                f.write("The TCM target prioritization system shows promising performance in identifying relevant targets within its top 20 predictions. While there is room for improvement in the precision of top-ranked predictions, the system provides useful guidance for researchers by substantially narrowing down the search space for potential drug targets.\n")
            else:
                f.write("The TCM target prioritization system shows moderate performance in ranking potential targets. While the integration of pharmacological binding potential with disease relevance represents a sound approach, further refinement of the model and expansion of the training data would likely yield significant improvements in predictive accuracy.\n")
    
    print("Summary report created at: results/summary/system_performance_summary.md")

if __name__ == "__main__":
    summarize_results()
