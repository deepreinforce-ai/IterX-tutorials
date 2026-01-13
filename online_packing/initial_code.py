import numpy as np
from numba import jit

@jit
def score(item, bins):
    """Naive Best Fit: prefer bins with least remaining space after placement."""
    return 1 / (bins - item + 1)
