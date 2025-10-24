import numpy as np
from typing import override
from interfaces.policy import Policy

class Policy_DeterministicGreedy(Policy):
    """
    Deterministic greedy policy implementation
    
    The greedy policy concept is described throughout Sutton & Barto 2nd edition,
    particularly in Chapter 2 (Section 2.3) and Chapter 3 (Section 3.1).
    """
    def __init__(self, Q: np.ndarray):
        self.Q = Q

    @override
    def action(self, state: int) -> int:
        # pick action with highest Q value
        return int(np.argmax(self.Q[state]))


    @override
    def action_prob(self, state: int, action: int) -> float:
        # greedy policy: 1 for best action, 0 for others
        greedy_action = int(np.argmax(self.Q[state]))
        return 1.0 if action == greedy_action else 0.0