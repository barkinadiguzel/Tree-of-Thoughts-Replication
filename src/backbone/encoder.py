import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.cnn = models.resnet18(weights=None)  
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_dim)
        
    def forward(self, x):
        return self.cnn(x)  
