import pandas as pd
import pickle
import json
import logging
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import os

# Configure logging for the module
logging.basicConfig(
    filename="logs/modeleval.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.
    Args:
        model_path (str): Path to the model file.
    Returns:
        Any: Loaded model object.
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def load_test_data(test_path: str) -> pd.DataFrame:
    """
    Load test data from CSV file.
    Args:
        test_path (str): Path to the test CSV file.
    Returns:
        pd.DataFrame: Loaded test data.
    """
    try:
        test_data = pd.read_csv(test_path)
        logging.info(f"Test data loaded from {test_path}")
        return test_data
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        raise

def evaluate_model(model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate the model on test data and calculate metrics.
    Args:
        model (Any): Trained model.
        test_data (pd.DataFrame): Test data with features and labels.
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    try:
        # Separate features and labels
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise

def save_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.
    Args:
        metrics (Dict[str, float]): Metrics dictionary.
        metrics_path (str): Path to save the metrics JSON file.
    """
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")
        raise

def main() -> None:
    """
    Main function to orchestrate model evaluation and reporting.
    """
    try:
        # Load trained model
        model = load_model("models/random_forest_model.pkl")
        # Load test data
        test_data = load_test_data("data/interim/test_bow.csv")
        # Evaluate model
        metrics = evaluate_model(model, test_data)
        # Save metrics to JSON file
        save_metrics(metrics, "reports/metrics.json")
        logging.info("Model evaluation pipeline completed successfully")
    except Exception as e:
        logging.error(f"Model evaluation pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()