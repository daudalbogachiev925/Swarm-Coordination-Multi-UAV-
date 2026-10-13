import numpy as np
import matplotlib.pyplot as plt
from src.agent import Agent
from src.task_allocator import hungarian_allocate
from src.collision import detect_collisions, separation_force
from src.comms import CommunicationGraph

class SwarmSimulator:
    def __init__(self, n_agents=5, n_targets=5, dt=0.1):
        np.random.seed(42)
        self.agents = [Agent(i, np.random.randn(3) * 10) for i in range(n_agents)]
        self.targets = np.random.randn(n_targets, 3) * 20
        self.dt = dt
        self.comm = CommunicationGraph(n_agents)

    def run(self, steps=200):
        hist = []
        for _ in range(steps):
            alloc = hungarian_allocate(self.agents, self.targets)
            for ai, ti in alloc:
                self.agents[ai].target = self.targets[ti]

            # separation
            for i in range(len(self.agents)):
                for j in range(len(self.agents)):
                    if i != j:
                        self.agents[i].target += separation_force(
                            self.agents[i], self.agents[j]) * 0.1

            for a in self.agents:
                a.step(self.dt)
            hist.append([a.position.copy() for a in self.agents])

            collisions = detect_collisions(self.agents)
            if collisions:
                print(f"Collisions: {collisions}")
        return hist

    def plot(self, hist):
        hist = np.array(hist)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        for i in range(hist.shape[1]):
            ax.plot(hist[:, i, 0], hist[:, i, 1], hist[:, i, 2], label=f"UAV {i}")
        ax.scatter(self.targets[:, 0], self.targets[:, 1], self.targets[:, 2],
                   c="red", marker="x", s=100, label="Targets")
        ax.legend()
        plt.title("Swarm Coordination Simulation")
        plt.show()
