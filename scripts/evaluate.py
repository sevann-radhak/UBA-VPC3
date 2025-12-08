import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import mlflow
import yaml
from src.models.model import VisionModel
from src.utils.helpers import load_config, get_device
from datasets import load_dataset
from src.data.dataset import CustomDataset, get_data_loaders
from src.data.preprocessing import get_val_transforms
from src.metrics.metrics import calculate_all_metrics, get_predictions


def main():
    config = load_config("config/config.yaml")
    device = get_device()

    mlflow.set_experiment(config['mlflow']['experiment_name'])
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    dataset = load_dataset(config['data']['dataset_name'])
    test_data = dataset['test']
    split_idx = len(test_data) // 2

    test_images = [test_data[i]['img'] for i in range(split_idx, len(test_data))]
    test_labels = [test_data[i]['label'] for i in range(split_idx, len(test_data))]

    test_dataset = CustomDataset(
        images=test_images,
        labels=test_labels,
        transform=get_val_transforms()
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    model = VisionModel(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes']
    )

    checkpoint = torch.load("checkpoints/checkpoint_epoch_19.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    y_true, y_pred = get_predictions(model, test_loader, device)
    metrics = calculate_all_metrics(y_true, y_pred)

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        
        print("Test Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

