"""Model and image preprocessing."""
import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

WEIGHTS = VGG16_Weights.DEFAULT
CLASS_NAMES = ("bo", "not_bo")


def build_model(pretrained: bool = True) -> nn.Module:
    """Return a VGG16 backbone with a binary classification head.

    1 logit: sigmoid(logit) = probability of class `not_bo`.
    """
    backbone = vgg16(weights=WEIGHTS if pretrained else None)
    backbone.classifier[6] = nn.Linear(backbone.classifier[6].in_features, 1)
    return backbone


def preprocess():
    return WEIGHTS.transforms()


def configure_trainable(model: nn.Module, phase: str) -> None:
    """Train head first; optionally fine-tune the final convolutional block."""
    model.requires_grad_(False)
    if phase == "head":
        for parameter in model.classifier[6].parameters():
            parameter.requires_grad_(True)
    elif phase == "fine_tune":
        for parameter in model.features[24:].parameters():
            parameter.requires_grad_(True)
        for parameter in model.classifier[6].parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(f"Unknown phase: {phase}")
