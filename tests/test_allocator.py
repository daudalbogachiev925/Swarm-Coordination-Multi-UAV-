import numpy as np
from src.agent import Agent
from src.task_allocator import hungarian_allocate

def test_allocation_unique():
    agents = [Agent(0, np.array([0, 0, 0])), Agent(1, np.array([10, 0, 0]))]
    targets = np.array([[1, 0, 0], [11, 0, 0]])
    alloc = hungarian_allocate(agents, targets)
    assert len(alloc) == 2
    assert alloc[0][1] != alloc[1][1]
