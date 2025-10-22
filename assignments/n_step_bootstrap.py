from typing import Iterable, Tuple
import gymnasium as gym
import numpy as np

from interfaces.policy import RandomPolicy, Policy
from interfaces.solver import Solver, Hyperparameters

from assignments.policy_deterministic_greedy import Policy_DeterministicGreedy

class EpsilonGreedyPolicy(Policy):
    """Epsilon-greedy policy for exploration"""
    def __init__(self, Q: np.ndarray, epsilon: float = 0.1):
        self.Q = Q
        self.epsilon = epsilon
        self.nA = Q.shape[1]
        
    def action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.nA)
        else:
            return np.argmax(self.Q[state])
    
    def action_prob(self, state: int, action: int) -> float:
        best_action = np.argmax(self.Q[state])
        if action == best_action:
            return 1 - self.epsilon + self.epsilon / self.nA
        else:
            return self.epsilon / self.nA

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
    
    # Iterate through all trajectories
    for traj in trajs:
        # Convert trajectory to list for easier indexing
        traj_list = list(traj)
        T = len(traj_list)  # Length of the trajectory
        
        # Process each step of the trajectory
        for t in range(T):
            # Calculate the n-step return
            # We need to look ahead n steps (or until terminal)
            tau = t - n + 1  # Time whose estimate is being updated
            
            if tau >= 0:
                # Calculate G (the n-step return)
                G = 0
                
                # Calculate the sum of discounted rewards
                for i in range(tau + 1, min(tau + n, T) + 1):
                    if i <= T:
                        # Get reward from previous transition
                        # traj_list[i-1] = (s_{i-1}, a_{i-1}, r_i, s_i)
                        if i - 1 < len(traj_list):
                            _, _, r, _ = traj_list[i - 1]
                            G += (gamma ** (i - tau - 1)) * r
                
                # Add the bootstrapped value if we haven't reached the end
                if tau + n < T:
                    # Get the state at time tau + n
                    _, _, _, s_next = traj_list[tau + n - 1]
                    G += (gamma ** n) * V[s_next]
                
                # Get the state at time tau
                s_tau, _, _, _ = traj_list[tau]
                
                # Update V for state s_tau
                V[s_tau] += alpha * (G - V[s_tau])

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
        # Initialize Q values to small random values to break ties
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.pi = Policy_DeterministicGreedy(self.Q)

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
        
        # Get hyperparameters
        gamma = self.hyperparameters.gamma
        alpha = self.hyperparameters.alpha
        n = self.hyperparameters.n
        
        # Initialize episode
        state, _ = self.env.reset()
        
        # Store trajectory  
        states = []
        actions = []
        rewards = []
        
        # Initial state
        states.append(state)
        
        # Initialize T to infinity
        T = float('inf')
        t = 0
        
        episode_G = 0.0
        
        # For off-policy learning, use epsilon-greedy policy for behavior
        # This ensures exploration while learning the greedy target policy
        epsilon = 0.1
        
        # Generate initial action using epsilon-greedy
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = np.argmax(self.Q[state])
        actions.append(action)
        
        # Episode loop
        while True:
            if t < T:
                # Take action and observe next state and reward
                next_state, reward, terminated, truncated, _ = self.env.step(actions[t])
                states.append(next_state)
                rewards.append(reward)
                episode_G += reward
                
                # If terminal state reached, set T
                if terminated or truncated:
                    T = t + 1
                else:
                    # Select next action using epsilon-greedy
                    if np.random.random() < epsilon:
                        next_action = np.random.randint(0, self.env.action_space.n)
                    else:
                        next_action = np.argmax(self.Q[next_state])
                    actions.append(next_action)
            
            # Time to update (tau is the time whose estimate is being updated)
            tau = t - n + 1
            
            if tau >= 0:
                # Calculate importance sampling ratio for off-policy
                rho = 1.0
                for i in range(tau + 1, min(tau + n, T)):
                    # For each action in the n-step trajectory
                    # Calculate the ratio of target policy (greedy) to behavior policy (epsilon-greedy)
                    best_action = np.argmax(self.Q[states[i]])
                    
                    # Target policy probability (greedy)
                    if actions[i] == best_action:
                        pi_prob = 1.0
                    else:
                        pi_prob = 0.0
                    
                    # Behavior policy probability (epsilon-greedy)
                    if actions[i] == best_action:
                        b_prob = 1.0 - epsilon + epsilon / self.env.action_space.n
                    else:
                        b_prob = epsilon / self.env.action_space.n
                    
                    if b_prob > 0:
                        rho *= pi_prob / b_prob
                    else:
                        rho = 0
                        break
                
                # Calculate n-step return G
                G = 0
                for i in range(tau, min(tau + n, T)):
                    G += (gamma ** (i - tau)) * rewards[i]
                
                # Add bootstrap value if not terminal
                if tau + n < T:
                    s_n = states[tau + n]
                    a_n = actions[tau + n]
                    G += (gamma ** n) * self.Q[s_n, a_n]
                
                # Update Q value with importance sampling
                s_tau = states[tau]
                a_tau = actions[tau]
                self.Q[s_tau, a_tau] += alpha * rho * (G - self.Q[s_tau, a_tau])
            
            # Check if we should terminate the episode  
            if tau == T - 1:
                break
                
            t += 1

        return episode_G
