# Portfolio copy (English)

## Dog Identity Recognition with Transfer Learning

A PyTorch computer vision experiment that adapts ImageNet-pretrained VGG16 to binary image identity recognition. The pipeline supports staged training (frozen feature extractor followed by partial fine-tuning), image augmentation, checkpointing, inference and validation metrics. Based on an NVIDIA Deep Learning Institute educational exercise; the standalone training code and CLI in this repository were independently rewritten.

**Tech:** Python · PyTorch · TorchVision · VGG16 · Transfer Learning · Computer Vision · Binary Classification

**Original notebook result:** 96.67% validation accuracy on 30 images. This is a small, potentially non-independent validation split, not a test-set result or evidence of real-world readiness. Reproduced performance for this cleaned implementation: not yet measured.

**What I learned:** Reusing pretrained representations on small datasets; staged layer freezing/unfreezing; carefully interpreting accuracy on imbalanced, small validation sets; preparing reproducible model pipelines.

**Suggested project card:** Transfer Learning for Dog Identity Recognition — Adapting pretrained VGG16 to recognize a specific subject, with staged fine-tuning and a reproducible PyTorch inference pipeline.

**Suggested tags:** Computer Vision · Deep Learning · Transfer Learning · PyTorch

**Assets to create after running:**
- `assets/projects/dog-identity/cover.webp`: visual of an example dog + model classification overlay, only using images with appropriate rights
- `assets/projects/dog-identity/training-curve.webp`: learning/validation curves generated from models/history.csv
- `assets/projects/dog-identity/prediction.webp`: a real inference screenshot, clearly labeled as a demonstration

No verified new-model performance, test results or fabricated screenshots should be published.
