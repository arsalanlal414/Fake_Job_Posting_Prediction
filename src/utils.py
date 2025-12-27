"""
Utility functions for the Fake Job Postings Prediction project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets
    
    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df


def display_basic_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Display basic information about the dataframe
    
    Args:
        df: pandas DataFrame
        name: Name to display
    """
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nMissing Percentage:\n{(df.isnull().sum() / len(df) * 100).round(2)}%")
    

def plot_missing_values(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Visualize missing values in the dataframe
    
    Args:
        df: pandas DataFrame
        figsize: Figure size tuple
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("No missing values found!")
        return
    
    plt.figure(figsize=figsize)
    missing.plot(kind='bar')
    plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Number of Missing Values', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_class_distribution(df: pd.DataFrame, target_col: str = 'fraudulent', 
                            figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the distribution of target variable
    
    Args:
        df: pandas DataFrame
        target_col: Name of target column
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    df[target_col].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
    axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Class (0=Legitimate, 1=Fraudulent)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xticklabels(['Legitimate', 'Fraudulent'], rotation=0)
    
    # Add count labels
    for i, v in enumerate(df[target_col].value_counts()):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    df[target_col].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                       colors=['green', 'red'], labels=['Legitimate', 'Fraudulent'])
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Class Distribution Statistics")
    print(f"{'='*60}")
    print(df[target_col].value_counts())
    print(f"\nClass Balance Ratio: {df[target_col].value_counts()[0] / df[target_col].value_counts()[1]:.2f}:1")
    print(f"Fraud Percentage: {df[target_col].sum() / len(df) * 100:.2f}%")


def save_predictions(test_ids: np.ndarray, predictions: np.ndarray, 
                    output_path: str = 'results/predictions.csv') -> None:
    """
    Save predictions in submission format
    
    Args:
        test_ids: Array of job IDs
        predictions: Array of predictions
        output_path: Output file path
    """
    submission = pd.DataFrame({
        'job_id': test_ids,
        'fraudulent': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted fraudulent: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
