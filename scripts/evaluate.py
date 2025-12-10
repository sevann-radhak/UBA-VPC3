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
    
    subset_size = config['data'].get('subset_size')
    if subset_size and subset_size > 0:
        test_subset = min(subset_size // 5, len(test_data))
        test_data = test_data.select(range(test_subset))
    
    split_idx = len(test_data) // 2
    image_size = config['model'].get('image_size', 160)

    test_images = list(test_data['img'][split_idx:])
    test_labels = list(test_data['label'][split_idx:])

    test_dataset = CustomDataset(
        images=test_images,
        labels=test_labels,
        transform=get_val_transforms(image_size=image_size)
    )

    from src.data.dataset import get_optimal_num_workers
    import os
    
    num_workers = get_optimal_num_workers()
    pin_memory = device.type == 'cuda' and torch.cuda.is_available()
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )

    import glob
    checkpoint_files = glob.glob("checkpoints/checkpoint_epoch_*.pt")
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoints found in checkpoints/ directory")
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"📥 Loading checkpoint: {latest_checkpoint}")

    model = VisionModel(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes']
    )

    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    y_true, y_pred = get_predictions(model, test_loader, device)
    metrics = calculate_all_metrics(y_true, y_pred)

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        
        print("Test Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

