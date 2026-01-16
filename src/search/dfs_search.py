from typing import List
from thoughts.thought_utils import State
from thoughts.thought_generator import ThoughtGenerator
from thoughts.state_evaluator import StateEvaluator

def tot_dfs(root_state: State,
            generator: ThoughtGenerator,
            evaluator: StateEvaluator,
            k: int = 4,
            depth: int = 4,
            vthres: float = -1e9):
    best_leaf = None
    best_score = float('-inf')

    def dfs(curr: State, level: int):
        nonlocal best_leaf, best_score
        if level >= depth:
            # evaluate leaf
            sc = evaluator.score_states([curr])[0]
            if sc > best_score:
                best_score = sc
                best_leaf = curr
            return
        # generate candidates
        cands = generator(curr.embedding)
        children = [curr.clone_with(emb) for emb in cands]
        scores = evaluator.score_states(children)
        # sort by score descending and explore
        sorted_idx = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        for i in sorted_idx:
            if scores[i] <= vthres:
                continue  # prune
            dfs(children[i], level + 1)

    dfs(root_state, 0)
    return best_leaf
