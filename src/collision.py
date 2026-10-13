import numpy as np

def detect_collisions(agents, safe_dist=2.0):
    pairs = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            d = np.linalg.norm(agents[i].position - agents[j].position)
            if d < safe_dist:
                pairs.append((i, j, d))
    return pairs

def separation_force(a, b, safe_dist=2.0, gain=1.0):
    diff = a.position - b.position
    d = np.linalg.norm(diff)
    if d < 1e-6:
        return np.zeros(3)
    if d < safe_dist:
        return gain * (safe_dist - d) * diff / d
    return np.zeros(3)
