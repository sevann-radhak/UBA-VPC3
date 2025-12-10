import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.helpers import load_config

def main():
    try:
        config = load_config("config/config.yaml")
        print("✅ Configuración cargada correctamente")
        print("\n📋 Configuración actual:")
        print(f"  Modelo: {config['model']['name']}")
        print(f"  Clases: {config['model']['num_classes']}")
        print(f"  Dataset: {config['data']['dataset_name']}")
        print(f"  Batch Size: {config['training']['batch_size']}")
        print(f"  Learning Rate: {config['training']['learning_rate']}")
        print(f"  Épocas: {config['training']['num_epochs']}")
        print(f"  Device: {config['training']['device']}")
        print(f"  MLflow Experiment: {config['mlflow']['experiment_name']}")
    except Exception as e:
        print(f"❌ Error cargando configuración: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())




