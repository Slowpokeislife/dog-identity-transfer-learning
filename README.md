# Dog Identity Recognition with Transfer Learning

**PyTorch · TorchVision · VGG16 · Binary Image Classification**

A clean, standalone computer-vision experiment that uses **transfer learning** to distinguish a particular dog (in this educational case study, **Bo**) from other images. It starts from a VGG16 model pretrained on ImageNet, trains a new binary head, optionally fine-tunes the final convolutional block, and exposes a simple CLI for inference.

## Why this experiment matters

Identifying an individual animal is different from recognizing the general category *dog*. This case study examines whether features learned from a large classification dataset can help when the target identity has very few labeled examples.

## Dataset and expected results

| Metric | Value | Interpretation |
|---|---:|---|
| Training images | 139 (19 Bo / 120 other) | Imbalanced and small |
| Validation images | 30 (10 Bo / 20 other) | Small holdout |
| Independent held-out test | Not available | Generalization is not established |
| Clean implementation results | Not measured yet | Run training before reporting performance |

**Important:** This clean implementation ensures proper batch-wise weighting and reports class-specific precision, recall and F1 for future runs. Results will vary with data, splits, random seeds and hardware.

## Repository structure

```text
.
├── src/dog_identity/
│   ├── __init__.py
│   ├── model.py           # pretrained model, preprocessing, freeze/unfreeze
│   ├── train.py           # two-stage training, validation, metrics, checkpoint
│   └── predict.py         # single-image inference
├── tests/test_metrics.py # lightweight metric sanity check
├── data/README.md        # expected local dataset layout and data provenance
├── docs/PORTFOLIO_COPY.md
├── demo/                 # reserved for your own screenshots
├── models/               # generated checkpoints and training logs (ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Requirements: Python 3.10+ and sufficient RAM/GPU memory for VGG16. The first training run downloads official pretrained VGG16 weights (approximately 528 MB); internet is needed for that first download. For GPU support, install a matching PyTorch/TorchVision build using the [official PyTorch installer](https://pytorch.org/get-started/locally/).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Supply the data

Place licensed images under `data/presidential_doggy_door/train/{bo,not_bo}` and `data/presidential_doggy_door/valid/{bo,not_bo}`. Keep both classes in both splits. Do not commit the dataset unless its rights and provenance permit redistribution. See [data/README.md](data/README.md).

## Train

```bash
python -m dog_identity.train --data-dir data/presidential_doggy_door --batch-size 8 --epochs-head 8 --epochs-fine-tune 2
```

The first stage freezes all pretrained layers and trains the new classification head. The second stage fine-tunes the final feature block plus the head using a much lower learning rate. The best checkpoint by validation loss is saved to `models/best_model.pt` and per-epoch metrics to `models/history.csv`.

## Predict

```bash
python -m dog_identity.predict --image path/to/your_image.jpg --checkpoint models/best_model.pt
```

Output fields: `prediction`, `probability_bo`, `probability_not_bo`, `threshold_not_bo`. Probabilities are uncalibrated neural-network scores; they are **not** a reliable measure of confidence on out-of-distribution images. The model always assigns one of two classes.

## Test

```bash
pytest -q
```

## Limitations and future work

- Dataset is very small and imbalanced. A high validation accuracy can be misleading.
- Verify there is no duplicate/near-duplicate leakage across splits and collect an independent test set.
- Benchmark against a pretrained feature extractor or smaller backbone (e.g., ResNet18).
- Report a confusion matrix and multiple random seeds on a larger licensed dataset.
- For deployment, add an unknown/reject class or calibrated abstention rule.
- This is an educational classification prototype, **not** an actual physical access-control product.

## Credits

Educational inspiration: [TorchVision models](https://pytorch.org/vision/stable/models.html). All third-party photographs require separate usage rights.

## Visual two-stage demo (five example photos)

The supplied example images are in `demo/images/`: `bo_23.jpg` (Bo), `132.jpg` and `happy_dog.jpg` (other dogs), `brown_bear.jpg` (bear), and `sleepy_cat.jpg` (cat). **Check image reuse rights before publishing these photos.**

![Expected-label explanation; not a prediction](demo/example-flow.png)

Run the pretrained ImageNet VGG16 detector and, **if a compatible trained checkpoint is available**, the downstream Bo classifier:

```powershell
python -m pip install -e .
python -m dog_identity.visual_demo --images-dir demo/images --checkpoint models/best_model.pt
```

Outputs `demo/output/pipeline-results.png` (visual prediction report) and `demo/output/results.json` (machine-readable results). Without a checkpoint, Stage 1 still runs, and Stage 2 explicitly says **NOT RUN**; results are never invented. Model weights are downloaded by TorchVision on first use.

The first stage checks whether the **top-1 ImageNet category** is among dog-breed indices 151–268; this is *image classification*, not bounding-box object detection, and may reject difficult images. Stage 2 is evaluated only if Stage 1 accepts the image as a dog. The labels in `demo/example-flow.png` are illustrative ground-truth categories and must not be presented as recorded predictions.

To train on your own properly licensed dataset, populate `data/presidential_doggy_door/{train,valid}/{bo,not_bo}` and run `python -m dog_identity.train`. The five demo photos are **not** an adequate independent evaluation set.