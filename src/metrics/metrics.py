import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f1(y_true, y_pred)
    }


def get_predictions(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)




