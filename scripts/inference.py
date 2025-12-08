import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from PIL import Image
from src.models.model import VisionModel
from src.utils.helpers import load_config, get_device
from src.data.preprocessing import get_val_transforms


def main():
    config = load_config("config/config.yaml")
    device = get_device()

    model = VisionModel(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes']
    )

    checkpoint = torch.load("checkpoints/checkpoint_epoch_19.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform = get_val_transforms()

    image_path = input("Enter image path: ")
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    print(f"Predicted class: {predicted.item()}")
    print(f"Confidence: {probabilities[0][predicted].item():.4f}")


if __name__ == "__main__":
    main()

