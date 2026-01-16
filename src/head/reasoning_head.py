import torch
import torch.nn as nn

class ReasoningHead(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, num_classes))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return self.head(emb)
