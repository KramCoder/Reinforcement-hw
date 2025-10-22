import numpy as np
from typing import override
from interfaces.policy import Policy

class Policy_DeterministicGreedy(Policy):
    def __init__(self, Q: np.ndarray[np.float64]):
        """
        Parameters:
        - Q (np.ndarray): Q function; numpy array shape of [nS,nA]
        """
        self.Q = Q

    @override
    def action(self, state: int) -> int:
        """
        Chooses the action that maximizes the Q function for the given state.

        Parameters:
            - state (int): state index

        Returns:
            - int: index of the action to take
        """

        ### TODO: Implement the action method ###
        # Return the action with the highest Q-value for this state
        # In case of ties, argmax returns the first occurrence
        return np.argmax(self.Q[state])


    @override
    def action_prob(self, state: int, action: int) -> float:
        """
        Returns the probability of taking the action if we are in the given state.

        Since this is a greedy policy, this will be a 1 or a 0.

        Parameters:
            - state (int): state index
            - action (int): action index

        Returns:
            - float: the probability of taking the action in the given state
        """

        ### TODO: Implement the action_prob method ###
        # For deterministic greedy policy, probability is 1.0 for the best action, 0.0 for others
        best_action = np.argmax(self.Q[state])
        
        # Handle ties - if there are multiple max values, split probability equally
        max_q_value = self.Q[state, best_action]
        max_actions = np.where(self.Q[state] == max_q_value)[0]
        
        if action in max_actions:
            return 1.0 / len(max_actions)
        else:
            return 0.0