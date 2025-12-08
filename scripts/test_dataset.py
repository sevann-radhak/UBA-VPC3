import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.utils.helpers import load_config

def main():
    try:
        config = load_config("config/config.yaml")
        dataset_name = config['data']['dataset_name']
        
        print(f"📥 Cargando dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        
        print("\n✅ Dataset cargado correctamente")
        print(f"\n📊 Información del dataset:")
        
        if 'train' in dataset:
            print(f"  Train: {len(dataset['train'])} muestras")
            if len(dataset['train']) > 0:
                print(f"  Ejemplo train keys: {dataset['train'][0].keys()}")
                if 'label' in dataset['train'][0]:
                    print(f"  Label ejemplo: {dataset['train'][0]['label']}")
        
        if 'test' in dataset:
            print(f"  Test: {len(dataset['test'])} muestras")
            if len(dataset['test']) > 0:
                print(f"  Ejemplo test keys: {dataset['test'][0].keys()}")
        
        if 'validation' in dataset:
            print(f"  Validation: {len(dataset['validation'])} muestras")
        
        print(f"\n📋 Estructura completa: {list(dataset.keys())}")
        
    except Exception as e:
        print(f"❌ Error cargando dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit(main())

