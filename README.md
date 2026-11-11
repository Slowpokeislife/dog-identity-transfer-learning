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