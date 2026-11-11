"""Inference on one image using a locally trained checkpoint."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from .model import build_model, preprocess


def predict(image_path: Path, checkpoint_path: Path, device_name: str = "auto") -> dict:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    # Only load checkpoints created by this repository or another trusted source.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "vgg16" or checkpoint.get("classes") != ["bo", "not_bo"]:
        raise ValueError("Invalid checkpoint metadata")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    with Image.open(image_path) as file:
        image = preprocess()(file.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        p_not_bo = torch.sigmoid(model(image).flatten())[0].item()
    threshold = float(checkpoint.get("threshold", .5))
    return {"prediction": "not_bo" if p_not_bo >= threshold else "bo",
            "probability_bo": round(1 - p_not_bo, 4),
            "probability_not_bo": round(p_not_bo, 4),
            "threshold_not_bo": threshold}


def main():
    parser = argparse.ArgumentParser(description="Predict if an image resembles the target dog")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    import json
    print(json.dumps(predict(args.image, args.checkpoint, args.device), indent=2))


if __name__ == "__main__":
    main()
