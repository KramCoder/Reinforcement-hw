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

    # Implement On-Policy n-Step TD Prediction (Sutton & Barto, p.144)
    # Iterate through all trajectories and perform TD updates on V
    V = np.array(initV, dtype=float)

    for traj in trajs:
        T = len(traj)
        if T == 0:
            continue

        # Reconstruct states and rewards arrays from the trajectory
        # Each element of traj is (s_t, a_t, r_{t+1}, s_{t+1})
        states = [transition[0] for transition in traj]
        states.append(traj[-1][3])  # s_T
        rewards = [transition[2] for transition in traj]

        # Perform n-step TD updates for each time step t
        for t in range(T):
            tau_end = min(t + n, T)

            # n-step return
            G = 0.0
            power = 0
            for k in range(t, tau_end):
                G += (gamma ** power) * rewards[k]
                power += 1

            if t + n < T:
                G += (gamma ** n) * V[states[t + n]]

            s_t = states[t]
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

        # Off-Policy n-Step SARSA (Sutton & Barto, p.149)
        hp: NStepSARSAHyperparameters = self.hyperparameters
        gamma = hp.gamma
        alpha = hp.alpha
        n = hp.n

        # Behavior policy (uniform random)
        bpi = RandomPolicy(self.env.action_space.n)

        # Generate one episode using the behavior policy
        states: list[int] = []
        actions: list[int] = []
        rewards: list[float] = []

        episode_G = 0.0
        state, _ = self.env.reset()
        done = False

        while not done:
            action = bpi.action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            # Log transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            episode_G += reward

            state = next_state
            done = terminated or truncated

        # Append terminal (or last) state
        states.append(state)

        T = len(rewards)
        if T == 0:
            return episode_G

        # Perform n-step SARSA updates with weighted importance sampling
        Q = self.pi.Q  # reference to policy's Q-table

        for t in range(T):
            tau_end = min(t + n, T)

            # Compute n-step return G
            G = 0.0
            power = 0
            for k in range(t, tau_end):
                G += (gamma ** power) * rewards[k]
                power += 1
            # Bootstrapping term uses the target policy's greedy action at time t+n
            if t + n < T:
                next_state = states[t + n]
                next_action = self.pi.action(next_state)
                G += (gamma ** n) * Q[next_state, next_action]

            # Importance sampling ratio ρ from k=t+1 to min(t+n, T-1)
            rho = 1.0
            for k in range(t + 1, tau_end):
                pi_prob = self.pi.action_prob(states[k], actions[k])
                b_prob = bpi.action_prob(states[k], actions[k])
                if pi_prob == 0.0:
                    rho = 0.0
                    break
                rho *= pi_prob / b_prob

            s_t = states[t]
            a_t = actions[t]
            Q[s_t, a_t] += alpha * rho * (G - Q[s_t, a_t])

        return episode_G
