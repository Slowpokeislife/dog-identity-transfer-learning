"""Visual two-stage inference on the five supplied example photos.

Stage 1: ImageNet pretrained VGG16, ImageNet dog indices 151..268.
Stage 2: Bo-vs-not-Bo classifier, only with a trained checkpoint.
This is image classification, not object localization/detection.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision.models import VGG16_Weights, vgg16

from .model import build_model

EXAMPLES = [
    ("brown_bear.jpg", "Bear", False, None),
    ("bo_23.jpg", "Bo", True, True),
    ("sleepy_cat.jpg", "Cat", False, None),
    ("132.jpg", "Other dog A", True, False),
    ("happy_dog.jpg", "Other dog B", True, False),
]
WEIGHTS = VGG16_Weights.DEFAULT

def image_tensor(path, device):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return WEIGHTS.transforms()(rgb).unsqueeze(0).to(device)

def load_identity(checkpoint_path, device):
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "vgg16" or checkpoint.get("classes") != ["bo", "not_bo"]:
        raise ValueError("Checkpoint metadata does not match the Bo classifier")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()

def run(images_dir: Path, checkpoint_path: Path, output_dir: Path, device_name="auto"):
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    missing = [name for name, *_ in EXAMPLES if not (images_dir / name).is_file()]
    if missing: raise FileNotFoundError("Missing demo photos: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = vgg16(weights=WEIGHTS).to(device).eval()
    identifier = load_identity(checkpoint_path, device)
    rows = []
    for filename, subject, expected_dog, expected_bo in EXAMPLES:
        tensor = image_tensor(images_dir / filename, device)
        with torch.inference_mode():
            scores = detector(tensor)[0].softmax(dim=0)
            idx = int(scores.argmax())
            dog = 151 <= idx <= 268  # ImageNet dog-breed synsets
            top1 = str(WEIGHTS.meta["categories"][idx])
            top1_score = float(scores[idx])
            bo_prob = None; bo_prediction = None
            if dog and identifier is not None:
                bo_prob = float(1.0 - torch.sigmoid(identifier(tensor).flatten())[0])
                bo_prediction = "Bo" if bo_prob > .5 else "Not Bo"
        rows.append({"image": filename, "subject": subject, "expected_dog": expected_dog,
                     "expected_bo": expected_bo, "stage1_predicted_dog": dog,
                     "stage1_top1_class": top1, "stage1_top1_probability": round(top1_score, 5),
                     "stage2_prediction": bo_prediction,
                     "stage2_probability_bo": round(bo_prob, 5) if bo_prob is not None else None,
                     "stage2_status": "not_run_no_checkpoint" if identifier is None else ("skipped_non_dog" if not dog else "run")})
        print(f"{subject:13s} | dog={str(dog):5s} | ImageNet={top1:24.24s} | identity={bo_prediction or 'not run'}")
    (output_dir / "results.json").write_text(json.dumps({"notes": ["Actual inference run; no numbers are simulated", "Dog detection uses ImageNet top-1 synsets, not bounding boxes", "Bo classifier is evaluated only on images accepted as dogs"], "results": rows}, indent=2), encoding="utf-8")
    render(rows, images_dir, output_dir / "pipeline-results.png")
    return rows

def render(rows, images_dir, target):
    # PIL-only static report. No charts or predictions are fabricated.
    W, gap, cell_w, thumb_h = 1500, 18, 270, 220
    H = 860
    image = Image.new("RGB", (W, H), "#0b111a")
    draw = ImageDraw.Draw(image)
    try:
        font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        fonts = {size: ImageFont.truetype(font_file, size) for size in (16,20,25,32,44)}
    except OSError:
        fonts = {size: ImageFont.load_default() for size in (16,20,25,32,44)}
    def txt(pos, content, size=20, color="#e9edf2"):
        draw.text(pos, str(content), font=fonts[size], fill=color)
    txt((42,24), "Transfer Learning | Visual Inference", 44)
    txt((44,88), "STAGE 1  /  ImageNet VGG16 dog classification", 25, "#ef923f")
    for i,row in enumerate(rows):
        x=42+i*(cell_w+gap); y=136
        draw.rounded_rectangle((x-3,y-3,x+cell_w+3,y+thumb_h+118),radius=12,fill="#162432",outline="#344455",width=2)
        with Image.open(images_dir/row["image"]) as src:
            photo=ImageOps.fit(src.convert("RGB"),(cell_w,thumb_h))
            image.paste(photo,(x,y))
        txt((x+12,y+thumb_h+8),row["subject"],20)
        txt((x+12,y+thumb_h+39),"DOG" if row["stage1_predicted_dog"] else "NOT DOG",20,"#77d5a3" if row["stage1_predicted_dog"] else "#ff9b8f")
        txt((x+12,y+thumb_h+74),row["stage1_top1_class"][:23],16,"#a8b6c6")
    txt((44,510),"STAGE 2  /  Bo vs Not Bo (only accepted dogs)",25,"#ef923f")
    dogs=[r for r in rows if r["expected_dog"]]
    for i,row in enumerate(dogs):
        x=42+i*(cell_w+gap);y=556
        draw.rounded_rectangle((x-3,y-3,x+cell_w+3,y+thumb_h+48),radius=12,fill="#162432",outline="#344455",width=2)
        with Image.open(images_dir/row["image"]) as src:
            image.paste(ImageOps.fit(src.convert("RGB"),(cell_w,thumb_h)),(x,y))
        stage=row["stage2_prediction"] or ("SKIPPED: Not dog" if not row["stage1_predicted_dog"] else "NOT RUN: No checkpoint")
        txt((x+10,y+thumb_h+9),stage[:23],16,"#ffc184")
    txt((940,585),"Expected labels are not",20,"#a8b6c6")
    txt((940,615),"model predictions.",20,"#a8b6c6")
    txt((940,672),"Generate real results with:",16,"#a8b6c6")
    txt((940,703),"python -m dog_identity.visual_demo",16,"#ffc184")
    txt((42,838),"Real predictions from this run | ImageNet top-1 breed filter | Stage 2 requires trained checkpoint",16,"#a8b6c6")
    image.save(target)

def main():
    parser=argparse.ArgumentParser(description="Generate five-photo two-stage inference report")
    parser.add_argument("--images-dir",type=Path,default=Path("demo/images"))
    parser.add_argument("--checkpoint",type=Path,default=Path("models/best_model.pt"))
    parser.add_argument("--output-dir",type=Path,default=Path("demo/output"))
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    args=parser.parse_args()
    run(args.images_dir,args.checkpoint,args.output_dir,args.device)
    print("Saved:",args.output_dir/"results.json", "and", args.output_dir/"pipeline-results.png")
if __name__ == "__main__": main()
