import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib
from ml_intention.features import build_windowed_dataset
import os

def generate_ml_diagnostics(dataset_path="imu_intention_dataset.csv", model_path="intention_model.joblib"):
    os.makedirs("outputs", exist_ok=True)
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(dataset_path)
        X, y = build_windowed_dataset(df)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # 1. Classification Report
        report = classification_report(y_test, y_pred)
        with open("outputs/ml_classification_report.txt", "w") as f:
            f.write(report)

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("outputs/ml_roc_curve.png")
        plt.close()
        
        print("ML Diagnostics generated in outputs/")
    except Exception as e:
        print(f"Skipping diagnostics: {e}")