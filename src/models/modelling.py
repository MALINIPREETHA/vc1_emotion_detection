import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
import os

# Configure logging for the module
logging.basicConfig(
    filename="logs/modelling.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_params(params_path: str) -> dict:
    """
    Load model parameters from a YAML file.
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

def load_training_data(train_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data and separate features and labels.
    Args:
        train_path (str): Path to training CSV file.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels arrays.
    """
    try:
        train_data = pd.read_csv(train_path)
        x_train = train_data.drop(columns=['label']).values  # Features
        y_train = train_data['label'].values                # Target labels
        logging.info(f"Training data loaded from {train_path}")
        return x_train, y_train
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        n_estimators (int): Number of trees.
        max_depth (int): Maximum tree depth.
    Returns:
        RandomForestClassifier: Trained model.
    """
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)
        logging.info("Random Forest model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """
    Save the trained model to disk using pickle.
    Args:
        model (RandomForestClassifier): Trained model.
        model_path (str): Path to save the model.
    """
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def main() -> None:
    """
    Main function to orchestrate model training and saving.
    """
    try:
        # Load parameters from YAML file
        params = load_params('params.yaml')
        n_estimators = params["modelling"]["n_estimators"]
        max_depth = params["modelling"]["max_depth"]

        # Load training data
        x_train, y_train = load_training_data("data/interim/train_bow.csv")

        # Train the Random Forest model
        model = train_model(x_train, y_train, n_estimators, max_depth)

        # Save the trained model to disk
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model training pipeline completed successfully")
    except Exception as e:
        logging.error(f"Model training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()