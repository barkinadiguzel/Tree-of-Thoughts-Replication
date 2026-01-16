from dataclasses import dataclass
import torch
from typing import List

@dataclass
class State:
    embedding: torch.Tensor
    past: List[torch.Tensor]
    depth: int = 0

    def clone_with(self, new_embedding: torch.Tensor):
        return State(
            embedding=new_embedding,
            past=self.past + [new_embedding],
            depth=self.depth + 1
        )

def to_device(state, device):
    emb = state.embedding.to(device)
    past = [p.to(device) for p in state.past]
    return State(embedding=emb, past=past, depth=state.depth)
