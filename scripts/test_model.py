import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models.model import VisionModel
from src.utils.helpers import load_config, get_device

def main():
    try:
        config = load_config("config/config.yaml")
        device = get_device()
        
        print(f"📥 Cargando modelo: {config['model']['name']}")
        print(f"🖥️  Device: {device}")
        
        model = VisionModel(
            model_name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            freeze_backbone=config['model']['freeze_backbone']
        )
        
        model = model.to(device)
        model.eval()
        
        print("✅ Modelo cargado correctamente")
        print(f"\n📊 Información del modelo:")
        print(f"  Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Probar forward pass
        print(f"\n🧪 Probando forward pass...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass exitoso")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output classes: {output.shape[1]}")
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit(main())

