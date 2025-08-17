import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging for the module
logging.basicConfig(
    filename="logs/featureengineering.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    Args:
        params_path (str): Path to the YAML file.
    Returns:
        dict: Parameters loaded from the file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data from CSV files.
    Args:
        train_path (str): Path to train CSV.
        test_path (str): Path to test CSV.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    """
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info(f"Train and test data loaded from {train_path} and {test_path}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to load train/test data: {e}")
        raise

def extract_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract Bag of Words features from train and test data.
    Args:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.
        max_features (int): Maximum number of features for vectorizer.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Feature DataFrames for train and test.
    """
    try:
        # Extract text and labels
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        # Initialize CountVectorizer
        vectorizer = CountVectorizer(max_features=max_features)
        # Fit and transform train data
        X_train_bow = vectorizer.fit_transform(X_train)
        # Transform test data
        X_test_bow = vectorizer.transform(X_test)

        # Convert to DataFrames and add labels
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logging.info("Feature extraction using Bag of Words completed")
        return train_df, test_df
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        raise

def save_features(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """
    Save feature DataFrames to CSV files.
    Args:
        train_df (pd.DataFrame): Train features.
        test_df (pd.DataFrame): Test features.
        output_dir (str): Directory to save CSV files.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train_bow.csv")
        test_path = os.path.join(output_dir, "test_bow.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logging.info(f"Feature data saved to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to save feature data: {e}")
        raise

def main() -> None:
    """
    Main function to orchestrate feature extraction steps.
    """
    try:
        # Load parameters from YAML file
        params = load_params('params.yaml')
        max_features = params["features"]["max_features"]

        # Load processed train and test data
        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")

        # Extract Bag of Words features
        train_df, test_df = extract_features(train_data, test_data, max_features)

        # Save the processed feature data to CSV files
        save_features(train_df, test_df, "data/interim")
        logging.info("Feature extraction pipeline completed successfully")
    except Exception as e:
        logging.error(f"Feature extraction pipeline failed: {e}")

if __name__ == "__main__":
    main()