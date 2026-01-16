# Search / thought params
EMBEDDING_DIM = 256
K = 4            # candidate thoughts per step
DEPTH = 3        # max depth (T)
BEAM = 4         # BFS beam width
VTHRES = -1e9    # DFS pruning threshold

# Model params
NUM_CLASSES = 2
DEVICE = "cpu"   # or "cuda" if available
