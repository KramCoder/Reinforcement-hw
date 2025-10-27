import gymnasium as gym
import numpy as np

class BanditEnvironment(gym.Env):
    """
    Represents a k-armed bandit problem as a gym environment.
    """

    def __init__(self, k: int):
        """
        Parameters:
            k (int): the number of bandit arms
        """
        self.action_space = gym.spaces.Discrete(k)
        self.observation_space = None # for bandits we don't use states
        self.Q_star = np.zeros((self.action_space.n))

    def step(self, a: int) -> tuple[float, bool]:
        """
        Performs a step of the bandit.

        Parameters:
            a (int): the selected action to draw a sample from.

        Returns:
            tuple: A tuple containing:
                - next_state (None): Always None for bandit problems, since there is no state.
                - reward (float): The reward of the selected bandit arm.
                - done (False): Always False for bandit problems, since there is no state.
                - truncated (False): False for bandits, although the TimeLimit wrapper will set this to True.
                - info (dict): A dictionary with the following keys:
                    - "ideal_action" (bool): True if the selected arm was one of the best
        """
        assert a < len(self.Q_star), f"Invalid action: {a}"
        reward = self._sample_reward(a)
        ideal_action = self._is_ideal_action(a)
        self._walk_all_arms()
        return None, reward, False, False, {"ideal_action": ideal_action}

    def reset(self) -> None:
        """
        Resets the bandit to its initial state.
        """
        self.Q_star = np.zeros_like(self.Q_star)

        return None, {}

    def _sample_reward(self, action: int) -> float:
        """
        Samples a reward from the selected arm. r ~ N(Q*(a), 1.0)

        Parameters:
            action (int): the selected action
        """
        # Sample a reward from a normal distribution centred at the true value
        # of the selected arm with standard deviation 1.0 (variance 1).
        return float(np.random.normal(loc=self.Q_star[action], scale=1.0))

    def _is_ideal_action(self, action: int) -> bool:
        """
        Determines if the selected action is one of the best actions.
        
        Parameters:
            action (int): the selected action
            
        Returns:
            bool: True if the selected action is one of the best actions.
        """
        # If the selected arm has the maximal Q* value (within numerical
        # tolerance) we treat it as an ideal action. There may be several such
        # arms, so we compare using a tolerance rather than exact equality.
        best_value = self.Q_star.max()
        return np.isclose(self.Q_star[action], best_value)


    def _walk_all_arms(self):
        """
        Walks all the arms of the bandit, adding noise to the true Q* values.

        noise ~ N(0, 0.01)
        """
        # Perform a random walk on all true action values by adding Gaussian
        # noise with mean 0 and standard deviation 0.01.
        self.Q_star += np.random.normal(loc=0.0, scale=0.01, size=self.Q_star.shape)
