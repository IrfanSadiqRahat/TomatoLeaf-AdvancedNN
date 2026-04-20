"""
Advanced neural network architectures for tomato leaf disease.
Published in Discover Sustainability (Springer), 2025.
DOI: 10.1007/s43621-025-01149-1
"""
import torch
import torch.nn as nn
import torchvision.models as tvm
from typing import Literal


ModelName = Literal["efficientnet_b3", "resnet50", "densenet121", "mobilenet_v3"]

def build_model(name: ModelName, num_classes: int = 10,
                pretrained: bool = True) -> nn.Module:
    """Factory for benchmark architectures used in the paper."""
    if name == "efficientnet_b3":
        m = tvm.efficientnet_b3(pretrained=pretrained)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif name == "resnet50":
        m = tvm.resnet50(pretrained=pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m = tvm.densenet121(pretrained=pretrained)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "mobilenet_v3":
        m = tvm.mobilenet_v3_large(pretrained=pretrained)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m
