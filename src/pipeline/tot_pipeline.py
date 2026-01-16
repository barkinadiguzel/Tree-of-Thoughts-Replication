import torch
from backbone.encoder import VisionEncoder
from thoughts.thought_utils import State
from thoughts.thought_generator import ThoughtGenerator
from thoughts.state_evaluator import ValueNetwork, StateEvaluator
from head.reasoning_model import ReasoningHead
from search.bfs_search import tot_bfs
from search.dfs_search import tot_dfs
from typing import Optional

class ToTPipeline:
    def __init__(self,
                 embedding_dim: int = 256,
                 k: int = 4,
                 depth: int = 3,
                 beam: int = 5,
                 num_classes: int = 2,
                 device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device('cpu')
        self.encoder = VisionEncoder(embedding_dim=embedding_dim).to(self.device)
        self.generator = ThoughtGenerator(embedding_dim=embedding_dim, k=k).to(self.device)
        self.value_net = ValueNetwork(embedding_dim=embedding_dim).to(self.device)
        self.evaluator = StateEvaluator(self.value_net, device=self.device)
        self.reasoning = ReasoningHead(embedding_dim=embedding_dim, num_classes=num_classes).to(self.device)
        self.k = k
        self.depth = depth
        self.beam = beam

    def forward(self, image_tensor: torch.Tensor, strategy: str = 'bfs'):
        self.encoder.eval()
        self.generator.eval()
        self.value_net.eval()
        self.reasoning.eval()

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            emb = self.encoder(image_tensor)   
            emb = emb.squeeze(0)              
            root = State(embedding=emb, past=[], depth=0)
            if strategy == 'bfs':
                best_state = tot_bfs(root, self.generator, self.evaluator,
                                     k=self.k, depth=self.depth, beam=self.beam)
            else:
                best_state = tot_dfs(root, self.generator, self.evaluator,
                                     k=self.k, depth=self.depth, vthres=float('-inf'))
            logits = self.reasoning(best_state.embedding.to(self.device))  
            return logits
