import numpy as np

def consensus_update(agents, adjacency, dt, alpha=0.5):
    """Средний консенсус по позиции (каждый агент тянется к соседям)."""
    for i, a in enumerate(agents):
        neighbors = [agents[j].position for j in range(len(agents)) if adjacency[i][j]]
        if not neighbors:
            continue
        target = np.mean(neighbors, axis=0)
        a.position = a.position + alpha * (target - a.position) * dt
    return agents
