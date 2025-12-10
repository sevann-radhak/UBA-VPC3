import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import mlflow
import yaml
from src.models.model import VisionModel
from src.models.trainer import Trainer
from src.utils.helpers import load_config, get_device, ensure_dir
from datasets import load_dataset
from src.data.dataset import CustomDataset, get_data_loaders
from src.data.preprocessing import get_train_transforms, get_val_transforms
from src.metrics.metrics import calculate_all_metrics, get_predictions


def main():
    config = load_config("config/config.yaml")
    device = get_device()
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    try:
        experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
        if experiment is None:
            experiment_id = mlflow.create_experiment(config['mlflow']['experiment_name'])
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        experiment_id = mlflow.create_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params({
            'model_name': config['model']['name'],
            'num_classes': config['model']['num_classes'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'num_epochs': config['training']['num_epochs']
        })

        print("📥 Loading dataset...")
        dataset = load_dataset(config['data']['dataset_name'])
        
        train_data = dataset['train']
        test_data = dataset['test']

        subset_size = config['data'].get('subset_size')
        fast_mode = config['training'].get('fast_mode', False)
        image_size = config['model'].get('image_size', 160)

        if subset_size and subset_size > 0:
            print(f"⚡ Fast mode: Using subset of {subset_size} training samples")
            train_data = train_data.select(range(min(subset_size, len(train_data))))
            test_subset = min(subset_size // 5, len(test_data))
            test_data = test_data.select(range(test_subset))
            print(f"   Using {test_subset} test samples")

        print("📊 Preparing datasets...")
        train_images = list(train_data['img'])
        train_labels = list(train_data['label'])

        test_images = list(test_data['img'])
        test_labels = list(test_data['label'])

        split_idx = len(test_data) // 2

        train_dataset = CustomDataset(
            images=train_images,
            labels=train_labels,
            transform=get_train_transforms(image_size=image_size, fast_mode=fast_mode)
        )

        val_dataset = CustomDataset(
            images=test_images[:split_idx],
            labels=test_labels[:split_idx],
            transform=get_val_transforms(image_size=image_size)
        )

        test_dataset = CustomDataset(
            images=test_images[split_idx:],
            labels=test_labels[split_idx:],
            transform=get_val_transforms(image_size=image_size)
        )

        device_str = str(device)
        print(f"🚀 Creating data loaders (device: {device_str})...")
        train_loader, val_loader, test_loader = get_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=config['training']['batch_size'],
            device=device_str
        )

        use_compile = config['training'].get('use_compile', False) and device.type == 'cuda'
        if use_compile:
            print("⚡ Model compilation enabled for faster inference")
        
        model = VisionModel(
            model_name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            freeze_backbone=config['model']['freeze_backbone'],
            use_compile=use_compile
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )

        use_amp = device.type == 'cuda'
        if use_amp:
            print("⚡ Mixed precision (FP16) enabled for faster training")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp
        )

        ensure_dir(Path("checkpoints"))

        for epoch in range(config['training']['num_epochs']):
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate()

            metrics_to_log = {
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }
            
            if 'precision' in val_metrics:
                metrics_to_log['val_precision'] = val_metrics['precision']
                metrics_to_log['val_recall'] = val_metrics['recall']
                metrics_to_log['val_f1_score'] = val_metrics['f1_score']
            
            mlflow.log_metrics(metrics_to_log, step=epoch)

            trainer.save_checkpoint(
                f"checkpoints/checkpoint_epoch_{epoch}.pt",
                epoch,
                {**train_metrics, **val_metrics}
            )

            print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")

        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()

