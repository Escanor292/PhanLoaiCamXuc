"""
Compare Multiple Experiments

This script compares results from multiple experiments and generates a comparison report.

Usage:
    python compare_experiments.py exp1_dir/ exp2_dir/ exp3_dir/
    python compare_experiments.py experiments/person_a/exp001/ experiments/person_b/exp001/
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
from tabulate import tabulate


def load_experiment_results(exp_dir):
    """
    Load experiment results from directory.
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        dict: Experiment configuration and results
    """
    exp_dir = Path(exp_dir)
    
    # Load training config
    config_path = exp_dir / 'training_config.json'
    if not config_path.exists():
        print(f"Warning: No training_config.json found in {exp_dir}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load results if available
    results_path = exp_dir / 'results.txt'
    results = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            content = f.read()
            # Parse results (simple parsing)
            for line in content.split('\n'):
                if 'Test Loss:' in line:
                    results['test_loss'] = float(line.split(':')[1].strip())
                elif 'Micro F1:' in line:
                    results['micro_f1'] = float(line.split(':')[1].strip())
                elif 'Macro F1:' in line:
                    results['macro_f1'] = float(line.split(':')[1].strip())
                elif 'Hamming Loss:' in line:
                    results['hamming_loss'] = float(line.split(':')[1].strip())
    
    return {
        'path': str(exp_dir),
        'name': config.get('experiment_name', exp_dir.name),
        'config': config,
        'results': results
    }


def compare_experiments(exp_dirs):
    """
    Compare multiple experiments.
    
    Args:
        exp_dirs: List of experiment directory paths
        
    Returns:
        pd.DataFrame: Comparison table
    """
    experiments = []
    
    for exp_dir in exp_dirs:
        exp_data = load_experiment_results(exp_dir)
        if exp_data:
            experiments.append(exp_data)
    
    if not experiments:
        print("No valid experiments found!")
        return None
    
    # Create comparison table
    comparison_data = []
    
    for exp in experiments:
        config = exp['config']
        results = exp['results']
        
        row = {
            'Experiment': exp['name'],
            'Path': exp['path'],
            'Model': config.get('model_name', 'N/A'),
            'Epochs': config.get('num_epochs', 'N/A'),
            'Batch Size': config.get('batch_size', 'N/A'),
            'Learning Rate': config.get('learning_rate', 'N/A'),
            'Dropout': config.get('dropout_rate', 'N/A'),
            'Best Epoch': config.get('best_epoch', 'N/A'),
            'Train Loss': config.get('train_loss', 'N/A'),
            'Val Loss': config.get('val_loss', 'N/A'),
            'Test Loss': results.get('test_loss', 'N/A'),
            'Micro F1': results.get('micro_f1', 'N/A'),
            'Macro F1': results.get('macro_f1', 'N/A'),
            'Hamming Loss': results.get('hamming_loss', 'N/A')
        }
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def print_comparison(df):
    """Print comparison table in a nice format."""
    if df is None or df.empty:
        print("No data to compare!")
        return
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON")
    print("="*100 + "\n")
    
    # Configuration comparison
    print("Configuration:")
    print("-" * 100)
    config_cols = ['Experiment', 'Model', 'Epochs', 'Batch Size', 'Learning Rate', 'Dropout']
    print(tabulate(df[config_cols], headers='keys', tablefmt='grid', showindex=False))
    
    # Results comparison
    print("\n\nResults:")
    print("-" * 100)
    results_cols = ['Experiment', 'Train Loss', 'Val Loss', 'Test Loss', 'Micro F1', 'Macro F1', 'Hamming Loss']
    print(tabulate(df[results_cols], headers='keys', tablefmt='grid', showindex=False))
    
    # Best model
    print("\n\nBest Models:")
    print("-" * 100)
    
    if 'Macro F1' in df.columns and df['Macro F1'].dtype in ['float64', 'float32']:
        best_macro_f1_idx = df['Macro F1'].idxmax()
        print(f"Best Macro F1: {df.loc[best_macro_f1_idx, 'Experiment']} ({df.loc[best_macro_f1_idx, 'Macro F1']:.4f})")
    
    if 'Micro F1' in df.columns and df['Micro F1'].dtype in ['float64', 'float32']:
        best_micro_f1_idx = df['Micro F1'].idxmax()
        print(f"Best Micro F1: {df.loc[best_micro_f1_idx, 'Experiment']} ({df.loc[best_micro_f1_idx, 'Micro F1']:.4f})")
    
    if 'Val Loss' in df.columns and df['Val Loss'].dtype in ['float64', 'float32']:
        best_val_loss_idx = df['Val Loss'].idxmin()
        print(f"Best Val Loss: {df.loc[best_val_loss_idx, 'Experiment']} ({df.loc[best_val_loss_idx, 'Val Loss']:.4f})")
    
    print("\n" + "="*100 + "\n")


def save_comparison(df, output_file):
    """Save comparison to CSV file."""
    if df is None or df.empty:
        print("No data to save!")
        return
    
    df.to_csv(output_file, index=False)
    print(f"✓ Comparison saved to: {output_file}")


def generate_markdown_report(df, output_file):
    """Generate markdown report."""
    if df is None or df.empty:
        print("No data to generate report!")
        return
    
    with open(output_file, 'w') as f:
        f.write("# Experiment Comparison Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Number of Experiments:** {len(df)}\n\n")
        
        f.write("## Configuration Comparison\n\n")
        config_cols = ['Experiment', 'Model', 'Epochs', 'Batch Size', 'Learning Rate', 'Dropout']
        f.write(df[config_cols].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Results Comparison\n\n")
        results_cols = ['Experiment', 'Train Loss', 'Val Loss', 'Test Loss', 'Micro F1', 'Macro F1', 'Hamming Loss']
        f.write(df[results_cols].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Best Models\n\n")
        
        if 'Macro F1' in df.columns and df['Macro F1'].dtype in ['float64', 'float32']:
            best_macro_f1_idx = df['Macro F1'].idxmax()
            f.write(f"- **Best Macro F1:** {df.loc[best_macro_f1_idx, 'Experiment']} ({df.loc[best_macro_f1_idx, 'Macro F1']:.4f})\n")
        
        if 'Micro F1' in df.columns and df['Micro F1'].dtype in ['float64', 'float32']:
            best_micro_f1_idx = df['Micro F1'].idxmax()
            f.write(f"- **Best Micro F1:** {df.loc[best_micro_f1_idx, 'Experiment']} ({df.loc[best_micro_f1_idx, 'Micro F1']:.4f})\n")
        
        if 'Val Loss' in df.columns and df['Val Loss'].dtype in ['float64', 'float32']:
            best_val_loss_idx = df['Val Loss'].idxmin()
            f.write(f"- **Best Val Loss:** {df.loc[best_val_loss_idx, 'Experiment']} ({df.loc[best_val_loss_idx, 'Val Loss']:.4f})\n")
        
        f.write("\n## Recommendations\n\n")
        
        # Simple recommendations based on results
        if 'Macro F1' in df.columns and df['Macro F1'].dtype in ['float64', 'float32']:
            best_idx = df['Macro F1'].idxmax()
            best_exp = df.loc[best_idx]
            
            f.write(f"Based on Macro F1 score, the best experiment is **{best_exp['Experiment']}** with:\n\n")
            f.write(f"- Learning Rate: {best_exp['Learning Rate']}\n")
            f.write(f"- Batch Size: {best_exp['Batch Size']}\n")
            f.write(f"- Dropout: {best_exp['Dropout']}\n")
            f.write(f"- Macro F1: {best_exp['Macro F1']:.4f}\n")
    
    print(f"✓ Markdown report saved to: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare multiple experiments')
    parser.add_argument('experiments', nargs='+', help='Paths to experiment directories')
    parser.add_argument('--output-csv', type=str, default='comparison.csv',
                        help='Output CSV file')
    parser.add_argument('--output-md', type=str, default='comparison.md',
                        help='Output Markdown file')
    parser.add_argument('--no-print', action='store_true',
                        help='Do not print to console')
    
    args = parser.parse_args()
    
    # Compare experiments
    df = compare_experiments(args.experiments)
    
    if df is None:
        return
    
    # Print comparison
    if not args.no_print:
        print_comparison(df)
    
    # Save comparison
    save_comparison(df, args.output_csv)
    generate_markdown_report(df, args.output_md)
    
    print("\n✓ Comparison complete!")
    print(f"  - CSV: {args.output_csv}")
    print(f"  - Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
