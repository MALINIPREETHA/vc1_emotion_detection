import os
import logging
from typing import Tuple
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Configure logging for the module
logging.basicConfig(
    filename="logs/dataingestion.log",
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

def load_dataset(url: str) -> pd.DataFrame:
    """
    Load dataset from a given URL.
    Args:
        url (str): URL to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(url)
        logging.info(f"Dataset loaded from {url}")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset: drop columns, filter, and encode labels.
    Args:
        df (pd.DataFrame): Raw dataset.
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    try:
        # Remove unnecessary columns
        df = df.drop(columns=['tweet_id'])
        # Filter for only 'happiness' and 'sadness'
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        # Encode sentiment labels to binary
        df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Data preprocessing completed")
        return df
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets.
    Args:
        df (pd.DataFrame): Preprocessed dataset.
        test_size (float): Proportion of test data.
        random_state (int): Random seed for reproducibility.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets.
    """
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train and test sets with test_size={test_size}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Data splitting failed: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> None:
    """
    Save train and test data to CSV files.
    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        output_dir (str): Directory to save CSV files.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'train.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Train and test data saved to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise

def main() -> None:
    """
    Main function to orchestrate data ingestion steps.
    """
    try:
        # Load parameters from YAML file
        params = load_params('params.yaml')
        test_size = params["dataingestion"]["test_size"]
        # Load dataset from URL
        url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = load_dataset(url)
        # Preprocess the dataset
        final_df = preprocess_data(df)
        # Split into train and test sets
        train_data, test_data = split_data(final_df, test_size)
        # Save the splits to disk
        save_data(train_data, test_data, 'data/raw')
        logging.info("Data ingestion pipeline completed successfully")
    except Exception as e:
        logging.error(f"Data ingestion pipeline failed: {e}")

if __name__ == "__main__":
    main()