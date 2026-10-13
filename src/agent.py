import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class Agent:
    id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    battery: float = 100.0

    def step(self, dt, kp=0.5):
        err = self.target - self.position
        self.velocity = kp * err
        self.position = self.position + self.velocity * dt
        self.battery -= 0.01 * np.linalg.norm(self.velocity) * dt
        return self.position
