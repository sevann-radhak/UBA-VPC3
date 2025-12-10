import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from PIL import Image
from src.models.model import VisionModel
from src.utils.helpers import load_config, get_device
from src.data.preprocessing import get_val_transforms


def main():
    try:
        config = load_config("config/config.yaml")
        device = get_device()
        
        import glob
        checkpoint_files = glob.glob("checkpoints/checkpoint_epoch_*.pt")
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoints found in checkpoints/ directory. Please train the model first.")
        
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

        image_size = config['model'].get('image_size', 160)
        transform = get_val_transforms(image_size=image_size)

        image_path = input("Enter image path: ")
        if not image_path:
            raise ValueError("Image path cannot be empty")
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Error loading image from {image_path}: {str(e)}")
        
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            probabilities = torch.nn.functional.softmax(output, dim=1)

        print(f"\n📊 Prediction Results:")
        print(f"  Predicted class: {predicted.item()}")
        print(f"  Confidence: {probabilities[0][predicted].item():.4f} ({probabilities[0][predicted].item()*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()




