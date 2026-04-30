#!/usr/bin/env python3
"""
CEFR Evaluation Script
Evaluate CEFR level prediction results (A1-C2, supporting intermediate levels 1.0-6.0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import sys

# Set matplotlib font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data(pred_file, gt_file):
    """
    Load prediction results and ground truth
    
    Args:
        pred_file: Path to prediction results CSV file
        gt_file: Path to ground truth CSV file
    
    Returns:
        pred_df, gt_df: Two DataFrames
    """
    try:
        pred_df = pd.read_csv(pred_file)
        gt_df = pd.read_csv(gt_file)
        return pred_df, gt_df
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

def validate_cefr_values(values, name):
    """Validate if CEFR values are within valid range"""
    valid_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    invalid = ~values.isin(valid_values)
    if invalid.any():
        print(f"Warning: {name} contains invalid CEFR values:")
        print(values[invalid].unique())
        return False
    return True

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary containing MSE, RMSE, MAE, R2
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def create_boxplot(y_true, y_pred, output_file='cefr_boxplot.png'):
    """
    Create boxplot: showing prediction score distribution for each true level
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_file: Output image file path
    """
    # Create DataFrame for grouping
    df = pd.DataFrame({
        'ground_truth': y_true,
        'prediction': y_pred
    })
    
    # Get all possible CEFR levels and sort
    levels = sorted(df['ground_truth'].unique())
    
    # Prepare predictions for each true level
    data_by_level = []
    labels = []
    
    # CEFR level mapping
    level_names = {
        1.0: 'A1', 1.5: 'A1+', 2.0: 'A2', 2.5: 'A2+',
        3.0: 'B1', 3.5: 'B1+', 4.0: 'B2', 4.5: 'B2+',
        5.0: 'C1', 5.5: 'C1+', 6.0: 'C2'
    }
    
    for level in levels:
        predictions_at_level = df[df['ground_truth'] == level]['prediction'].values
        data_by_level.append(predictions_at_level)
        # Label format: A1 (1.0)
        label = f"{level_names.get(level, str(level))} ({level})"
        labels.append(label)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Draw boxplot
    bp = ax.boxplot(data_by_level, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # Beautify boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.5, linestyle=':')
    
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=2)
    
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    
    for mean in bp['means']:
        mean.set(color='green', linewidth=2)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('Ground Truth CEFR Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted CEFR Score', fontsize=12, fontweight='bold')
    ax.set_title('CEFR Prediction Distribution by True Level', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis range
    ax.set_ylim(0.5, 6.5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='green', linewidth=2, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nBoxplot saved to: {output_file}")
    
    # Display figure (optional)
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description='CEFR Evaluation Script')
    parser.add_argument('pred_file', help='Path to prediction results CSV file')
    parser.add_argument('gt_file', help='Path to ground truth CSV file')
    parser.add_argument('--pred_col', default='prediction', help='Prediction column name (default: prediction)')
    parser.add_argument('--gt_col', default='ground_truth', help='Ground truth column name (default: ground_truth)')
    parser.add_argument('--output', default='cefr_boxplot.png', help='Boxplot output filename (default: cefr_boxplot.png)')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    pred_df, gt_df = load_data(args.pred_file, args.gt_file)
    
    # Check if columns exist
    if args.pred_col not in pred_df.columns:
        print(f"Error: Column '{args.pred_col}' not found in prediction file")
        print(f"Available columns: {list(pred_df.columns)}")
        sys.exit(1)
    
    if args.gt_col not in gt_df.columns:
        print(f"Error: Column '{args.gt_col}' not found in ground truth file")
        print(f"Available columns: {list(gt_df.columns)}")
        sys.exit(1)
    
    # Extract data
    y_pred = pred_df[args.pred_col].values
    y_true = gt_df[args.gt_col].values
    
    # Check data length
    if len(y_pred) != len(y_true):
        print(f"Error: Number of predictions ({len(y_pred)}) does not match number of ground truth values ({len(y_true)})")
        sys.exit(1)
    
    # Validate CEFR values
    print("\nValidating data...")
    validate_cefr_values(pd.Series(y_true), "Ground truth")
    validate_cefr_values(pd.Series(y_pred), "Predictions")
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print results
    print("\n" + "="*50)
    print("CEFR Evaluation Results")
    print("="*50)
    print(f"Number of samples: {len(y_true)}")
    print(f"\nMSE  (Mean Squared Error):       {metrics['MSE']:.4f}")
    print(f"RMSE (Root Mean Squared Error):  {metrics['RMSE']:.4f}")
    print(f"MAE  (Mean Absolute Error):      {metrics['MAE']:.4f}")
    print(f"R²   (R-squared):                {metrics['R2']:.4f}")
    print("="*50)
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print(f"Exact match accuracy: {(y_pred == y_true).mean()*100:.2f}%")
    print(f"Accuracy within ±0.5 level: {(np.abs(y_pred - y_true) <= 0.5).mean()*100:.2f}%")
    print(f"Accuracy within ±1.0 level: {(np.abs(y_pred - y_true) <= 1.0).mean()*100:.2f}%")
    
    # Create boxplot
    print("\nGenerating boxplot...")
    create_boxplot(y_true, y_pred, args.output)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()