import numpy as np
from typing import override
from interfaces.policy import Policy

class Policy_DeterministicGreedy(Policy):
    """
    Deterministic greedy policy: π(s) = argmax_a Q(s,a).

    Citation: Sutton & Barto, Reinforcement Learning (2nd ed.),
    Chapter 4, "Greedy policy with respect to action-value" (policy improvement).
    """
    def __init__(self, Q: np.ndarray):
        self.Q = Q

    @override
    def action(self, state: int) -> int:
        # greedy action selection w.r.t. Q(s,·) (SB2 Ch.4)
        return int(np.argmax(self.Q[state]))


    @override
    def action_prob(self, state: int, action: int) -> float:
        # π(a|s)=1 if a is greedy, else 0 (SB2 Ch.4)
        greedy_action = int(np.argmax(self.Q[state]))
        return 1.0 if action == greedy_action else 0.0