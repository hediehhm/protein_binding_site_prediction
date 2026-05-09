# pooling.py

import numpy as np


# -----------------------------------------------------
# MEAN POOLING (PRIMARY METHOD)
# -----------------------------------------------------

def mean_pool_embedding(token_embeddings: np.ndarray) -> np.ndarray:
    """
    Mean pooling over token-level embeddings to obtain a
    single fixed-size molecular embedding.

    Args:
        token_embeddings (np.ndarray):
            Shape: [num_tokens, hidden_dim]

    Returns:
        np.ndarray:
            Shape: [hidden_dim]
    """

    if token_embeddings is None:
        return None

    if len(token_embeddings) == 0:
        return None

    return np.mean(token_embeddings, axis=0)


