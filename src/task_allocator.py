import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_allocate(agents, targets):
    """Оптимальное распределение целей между агентами (Hungarian)."""
    n, m = len(agents), len(targets)
    if n == 0 or m == 0:
        return []
    cost = np.zeros((n, m))
    for i, a in enumerate(agents):
        for j, t in enumerate(targets):
            cost[i, j] = np.linalg.norm(a.position - t)
    rows, cols = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(rows, cols)]
