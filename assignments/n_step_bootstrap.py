from typing import Iterable, Tuple
import gymnasium as gym
import numpy as np

from interfaces.policy import RandomPolicy
from interfaces.solver import Solver, Hyperparameters

from assignments.policy_deterministic_greedy import Policy_DeterministicGreedy

def on_policy_n_step_td(
    trajs: Iterable[Iterable[Tuple[int,int,int,int]]],
    n: int,
    alpha: float,
    initV: np.array,
    gamma: float = 1.0
) -> Tuple[np.array]:
    """
    Runs the on-policy n-step TD algorithm to estimate the value function for a given policy.

    Sutton & Barto, p. 144, "n-step TD Prediction"

    Parameters:
        trajs (list): N trajectories generated using an unknown policy. Each trajectory is a 
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n (int): The number of steps (the "n" in n-step TD)
        alpha (float): The learning rate
        initV (np.ndarray): initial V values; np array shape of [nS]
        gamma (float): The discount factor

    Returns:
        V (np.ndarray): $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    V = np.array(initV)
    
    # Process each trajectory
    for traj in trajs:
        traj = list(traj)  # Convert to list for easier indexing
        T = len(traj)  # Number of transitions (not states)
        
        # For each time step t in the trajectory
        for t in range(T):
            # Calculate the n-step return G_t:t+n
            G = 0.0
            # Sum discounted rewards from t+1 to t+n
            for k in range(1, min(n + 1, T - t + 1)):
                if t + k - 1 < T:
                    _, _, r_t_k, _ = traj[t + k - 1]  # r_{t+k}
                    G += (gamma ** (k - 1)) * r_t_k
            
            # Add discounted value of state at t+n if we haven't reached the end
            if t + n < T:
                _, _, _, s_t_n = traj[t + n]  # s_{t+n}
                G += (gamma ** n) * V[s_t_n]
            # If t + n >= T, we don't bootstrap (episode ended)
            
            # Update V(S_t)
            s_t, _, _, _ = traj[t]  # s_t
            V[s_t] += alpha * (G - V[s_t])

    return V


class NStepSARSAHyperparameters(Hyperparameters):
    """ Hyperparameters for NStepSARSA algorithm """
    def __init__(self, gamma: float, alpha: float, n: int):
        """
        Parameters:
            gamma (float): The discount factor
            alpha (float): The learning rate
            n (int): The number of steps (the "n" in n-step SARSA)
        """
        super().__init__(gamma)
        self.alpha = alpha
        """The learning rate"""
        self.n = n
        """The number of steps (the "n" in n-step SARSA)"""

class NStepSARSA(Solver):
    """
    Solver for N-Step SARSA algorithm, good for discrete state and action spaces.

    Off-policy algorithm, using weighted importance sampling.
    """
    def __init__(self, env: gym.Env, hyperparameters: NStepSARSAHyperparameters):
        super().__init__("NStepSARSA", env, hyperparameters)
        self.pi = Policy_DeterministicGreedy(np.ones((env.observation_space.n, env.action_space.n)))

    def action(self, state):
        """
        Chooses an action based on the current policy.

        Parameters:
            state (int): The current state
        
        Returns:
            int: The action to take
        """
        return self.pi.action(state)

    def train_episode(self):
        """
        Trains the agent for a single episode.

        Returns:
            float: The total (undiscounted) reward for the episode
        """

        #####################
        # TODO: Implement Off Policy n-Step SARSA algorithm
        #   - Hint: Sutton Book p. 149
        #   - Hint: You'll need to build your trajectories using a behavior policy (RandomPolicy)
        #   - Hint: You can use the `pi.action_prob(state, action)` and `bpi.action_prob(state, action)` methods to get the action probabilities.
        #   - Hint: Be sure to check both terminated and truncated variables.
        #####################
        
        # Initialize behavior policy (random policy)
        bpi = RandomPolicy(self.env.action_space.n)
        
        # Generate episode using behavior policy
        state, _ = self.env.reset()
        episode = []
        episode_G = 0.0
        
        # Generate full episode
        while True:
            action = bpi.action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward, next_state))
            episode_G += reward
            
            if terminated or truncated:
                break
            state = next_state
        
        T = len(episode)  # Number of transitions
        
        # For each time step t in the episode
        for t in range(T):
            # Calculate the n-step return G_t:t+n
            G = 0.0
            # Sum discounted rewards from t+1 to t+n
            for k in range(1, min(self.hyperparameters.n + 1, T - t + 1)):
                if t + k - 1 < T:
                    _, _, r_t_k, _ = episode[t + k - 1]  # r_{t+k}
                    G += (self.hyperparameters.gamma ** (k - 1)) * r_t_k
            
            # Add discounted Q-value of state-action at t+n if we haven't reached the end
            if t + self.hyperparameters.n < T:
                s_t_n, a_t_n, _, _ = episode[t + self.hyperparameters.n]  # s_{t+n}, a_{t+n}
                G += (self.hyperparameters.gamma ** self.hyperparameters.n) * self.pi.Q[s_t_n, a_t_n]
            # If t + n >= T, we don't bootstrap (episode ended)
            
            # Calculate importance sampling ratio
            rho = 1.0
            for k in range(min(self.hyperparameters.n, T - t)):
                if t + k < T:
                    s_k, a_k, _, _ = episode[t + k]
                    pi_prob = self.pi.action_prob(s_k, a_k)
                    b_prob = bpi.action_prob(s_k, a_k)
                    if b_prob > 0:
                        rho *= pi_prob / b_prob
                    else:
                        rho = 0.0
                        break
            
            # Update Q(S_t, A_t)
            s_t, a_t, _, _ = episode[t]  # s_t, a_t
            self.pi.Q[s_t, a_t] += self.hyperparameters.alpha * rho * (G - self.pi.Q[s_t, a_t])

        return episode_G
