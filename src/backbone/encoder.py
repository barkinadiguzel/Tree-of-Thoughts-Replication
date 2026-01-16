import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512, backbone='resnet18'):
        super().__init__()
        if backbone == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)

    def forward(self, x):
        return self.model(x) 
