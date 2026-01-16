import torch
import torch.nn as nn
from typing import List, Dict
from .thought_utils import State

class ValueNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, 1)
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return self.net(emb).squeeze(-1)  

class StateEvaluator:
    def __init__(self, value_net: ValueNetwork, device=None):
        self.value_net = value_net
        self.device = device

    @torch.no_grad()
    def score_states(self, states: List[State]) -> Dict[int, float]:
        if len(states) == 0:
            return {}
        embeddings = torch.stack([s.embedding.detach().to(self.device) for s in states], dim=0)  
        vals = self.value_net(embeddings).cpu()  
        scores = {i: float(vals[i].item()) for i in range(len(states))}
        return scores

    @torch.no_grad()
    def vote_select(self, states: List[State], topk: int = 1) -> List[int]:
        scores = self.score_states(states)
        sorted_idx = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return sorted_idx[:topk]
