import torch
from torchvision import models
import config

def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.to(config.DEVICE)
    model.eval()
    return model