from typing import List
from thoughts.thought_utils import State
from thoughts.thought_generator import ThoughtGenerator
from thoughts.state_evaluator import StateEvaluator

def tot_bfs(root_state: State,
            generator: ThoughtGenerator,
            evaluator: StateEvaluator,
            k: int = 4,
            depth: int = 3,
            beam: int = 5):
    frontier: List[State] = [root_state]
    for t in range(depth):
        all_children: List[State] = []
        for s in frontier:
            cand_embs = generator(s.embedding)  
            for emb in cand_embs:
                child = s.clone_with(emb)
                all_children.append(child)
        if len(all_children) == 0:
            break
        scores = evaluator.score_states(all_children)  
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        selected = [all_children[i] for i in sorted_indices[:beam]]
        frontier = selected
    final_scores = evaluator.score_states(frontier)
    best_idx = max(final_scores.keys(), key=lambda i: final_scores[i])
    return frontier[best_idx]
