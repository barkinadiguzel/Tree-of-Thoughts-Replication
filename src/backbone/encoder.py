import torch
import torch.nn as nn
import torchvision.models as models

class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  
        self.backbone = backbone
        self.project = nn.Linear(in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)           
        emb = self.project(feats)           
        return emb
