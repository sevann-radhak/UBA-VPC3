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

        dataset = load_dataset(config['data']['dataset_name'])
        
        train_data = dataset['train']
        test_data = dataset['test']

        train_images = [train_data[i]['img'] for i in range(len(train_data))]
        train_labels = [train_data[i]['label'] for i in range(len(train_data))]

        test_images = [test_data[i]['img'] for i in range(len(test_data))]
        test_labels = [test_data[i]['label'] for i in range(len(test_data))]

        split_idx = len(test_data) // 2

        train_dataset = CustomDataset(
            images=train_images,
            labels=train_labels,
            transform=get_train_transforms()
        )

        val_dataset = CustomDataset(
            images=test_images[:split_idx],
            labels=test_labels[:split_idx],
            transform=get_val_transforms()
        )

        test_dataset = CustomDataset(
            images=test_images[split_idx:],
            labels=test_labels[split_idx:],
            transform=get_val_transforms()
        )

        train_loader, val_loader, test_loader = get_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=config['training']['batch_size']
        )

        model = VisionModel(
            model_name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            freeze_backbone=config['model']['freeze_backbone']
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        ensure_dir(Path("checkpoints"))

        for epoch in range(config['training']['num_epochs']):
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate()

            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }, step=epoch)

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

