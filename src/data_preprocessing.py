"""
Data preprocessing functions for Fake Job Postings Prediction
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def clean_text(text: str) -> str:
    """
    Clean text data by removing special characters and extra whitespace
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters and digits (optional - keep for now)
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def handle_missing_values(df: pd.DataFrame, strategy: str = 'fill') -> pd.DataFrame:
    """
    Handle missing values in the dataframe
    
    Args:
        df: Input dataframe
        strategy: Strategy to handle missing values ('fill', 'drop', 'indicator')
        
    Returns:
        Dataframe with handled missing values
    """
    df_copy = df.copy()
    
    # Text columns - fill with empty string
    text_columns = ['title', 'location', 'department', 'salary_range', 
                   'company_profile', 'description', 'requirements', 'benefits']
    
    for col in text_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('')
    
    # Categorical columns - fill with 'Unknown'
    cat_columns = ['employment_type', 'required_experience', 'required_education', 
                  'industry', 'function']
    
    for col in cat_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('Unknown')
    
    # Binary columns - fill with 0 (False)
    binary_columns = ['telecommuting', 'has_company_logo', 'has_questions']
    
    for col in binary_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(0)
    
    return df_copy


def create_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic features from text columns
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with new text features
    """
    df_copy = df.copy()
    
    # Length features
    df_copy['title_length'] = df_copy['title'].apply(lambda x: len(str(x)))
    df_copy['description_length'] = df_copy['description'].apply(lambda x: len(str(x)))
    df_copy['requirements_length'] = df_copy['requirements'].apply(lambda x: len(str(x)))
    df_copy['benefits_length'] = df_copy['benefits'].apply(lambda x: len(str(x)))
    df_copy['company_profile_length'] = df_copy['company_profile'].apply(lambda x: len(str(x)))
    
    # Word count features
    df_copy['title_word_count'] = df_copy['title'].apply(lambda x: len(str(x).split()))
    df_copy['description_word_count'] = df_copy['description'].apply(lambda x: len(str(x).split()))
    df_copy['requirements_word_count'] = df_copy['requirements'].apply(lambda x: len(str(x).split()))
    
    # Presence indicators
    df_copy['has_salary'] = df_copy['salary_range'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)
    df_copy['has_department'] = df_copy['department'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)
    df_copy['has_company_profile'] = df_copy['company_profile'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)
    df_copy['has_requirements'] = df_copy['requirements'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)
    df_copy['has_benefits'] = df_copy['benefits'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)
    
    # Special character count (might indicate spam)
    df_copy['description_special_chars'] = df_copy['description'].apply(
        lambda x: len(re.findall(r'[!@#$%^&*()]', str(x)))
    )
    
    # Capital letter ratio
    df_copy['title_capital_ratio'] = df_copy['title'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
    )
    
    return df_copy


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to clean the entire dataframe
    
    Args:
        df: Input dataframe
        
    Returns:
        Cleaned dataframe
    """
    print("Starting data cleaning...")
    
    # Make a copy
    df_clean = df.copy()
    
    # Handle missing values
    print("Handling missing values...")
    df_clean = handle_missing_values(df_clean)
    
    # Clean text columns
    print("Cleaning text columns...")
    text_cols = ['title', 'location', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_text)
    
    # Create text features
    print("Creating text features...")
    df_clean = create_text_features(df_clean)
    
    print(f"Cleaning complete! Shape: {df_clean.shape}")
    
    return df_clean


def get_feature_columns(exclude_cols: List[str] = None) -> Dict[str, List[str]]:
    """
    Get categorized feature columns
    
    Args:
        exclude_cols: Columns to exclude
        
    Returns:
        Dictionary of feature categories
    """
    if exclude_cols is None:
        exclude_cols = ['job_id', 'fraudulent']
    
    feature_groups = {
        'text_columns': ['title', 'location', 'company_profile', 'description', 
                        'requirements', 'benefits'],
        'categorical_columns': ['employment_type', 'required_experience', 
                               'required_education', 'industry', 'function'],
        'binary_columns': ['telecommuting', 'has_company_logo', 'has_questions',
                          'has_salary', 'has_department', 'has_company_profile',
                          'has_requirements', 'has_benefits'],
        'numeric_columns': ['title_length', 'description_length', 'requirements_length',
                           'benefits_length', 'company_profile_length', 'title_word_count',
                           'description_word_count', 'requirements_word_count',
                           'description_special_chars', 'title_capital_ratio']
    }
    
    return feature_groups
