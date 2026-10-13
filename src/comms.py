import numpy as np

class CommunicationGraph:
    def __init__(self, n_agents, comm_range=50):
        self.n = n_agents
        self.range = comm_range

    def adjacency(self, agents):
        A = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                d = np.linalg.norm(agents[i].position - agents[j].position)
                if d <= self.range:
                    A[i, j] = 1
        return A
