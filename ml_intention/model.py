import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ml_intention.features import build_windowed_dataset

class IntentionPredictor:
    """
    Wrapper for an intention classification model.
    Defaults to a Random Forest Classifier.
    """
    def __init__(self, model=None):
        if model is None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trains the underlying model."""
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probabilities of intent=1 (crossing) for each row in X.
        Assumes the positive class is at index 1.
        """
        probs = self.model.predict_proba(X)
        if probs.shape[1] > 1:
            return probs[:, 1]
        # Handle edge case where only class 0 is present
        return np.zeros(probs.shape[0])

    def save(self, path: str):
        """Serializes the model to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "IntentionPredictor":
        """Loads a serialized model from disk."""
        model = joblib.load(path)
        return cls(model=model)


def train_from_csv(csv_path: str,
                   model_path: str = "intention_model.joblib",
                   window_size: int = 10,
                   step_size: int = 5,
                   test_size: float = 0.2,
                   random_state: int = 42):
    """
    Loads IMU data, extracts features, trains a Random Forest model,
    prints evaluation metrics, and saves the trained model.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Building windowed dataset (window_size={window_size}, step_size={step_size})...")
    X, y = build_windowed_dataset(df, window_size=window_size, step_size=step_size)
    
    if len(X) == 0:
        print("Error: No valid windows extracted. Check window size and dataset length.")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Class distribution analysis
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    print("\nClass Distribution (Intent=0, Intent=1):")
    print(f"Train: {train_counts}")
    print(f"Test:  {test_counts}")

    print("\nTraining model...")
    predictor = IntentionPredictor()
    predictor.fit(X_train, y_train)

    # Evaluation
    y_pred = predictor.model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # Persistence
    predictor.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_from_csv("imu_intention_dataset.csv")