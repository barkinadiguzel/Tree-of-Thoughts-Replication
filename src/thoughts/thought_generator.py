import torch
import torch.nn as nn
from typing import List
from .thought_utils import State

class ThoughtGenerator(nn.Module):
    def __init__(self, embedding_dim: int = 256, k: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.k = k
        self.propose = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim * k)
        )

    def forward(self, state_embedding: torch.Tensor) -> List[torch.Tensor]:
        if state_embedding.dim() == 1:
            x = state_embedding.unsqueeze(0)  
        else:
            x = state_embedding       
        out = self.propose(x)       
        out = out.view(-1, self.k, self.embedding_dim)  
        out = out.squeeze(0)       
        return [out[i] for i in range(self.k)]
