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
    n-step TD algorithm implementation
    """

    # copy initial values
    V = np.array(initV, dtype=float)

    for traj in trajs:
        T = len(traj)
        if T == 0:
            continue

        # get states and rewards from trajectory
        states = [transition[0] for transition in traj]
        states.append(traj[-1][3])  # add final state
        rewards = [transition[2] for transition in traj]

        # do n-step updates
        for t in range(T):
            tau_end = min(t + n, T)

            # calculate n-step return
            G = 0.0
            for k in range(t, tau_end):
                G += (gamma ** (k - t)) * rewards[k]

            # add bootstrap value if not at end
            if t + n < T:
                G += (gamma ** n) * V[states[t + n]]

            # update V
            s_t = states[t]
            V[s_t] += alpha * (G - V[s_t])

    return V


class NStepSARSAHyperparameters(Hyperparameters):
    def __init__(self, gamma: float, alpha: float, n: int):
        super().__init__(gamma)
        self.alpha = alpha
        self.n = n

class NStepSARSA(Solver):
    """
    N-Step SARSA implementation
    """
    def __init__(self, env: gym.Env, hyperparameters: NStepSARSAHyperparameters):
        super().__init__("NStepSARSA", env, hyperparameters)
        self.pi = Policy_DeterministicGreedy(np.ones((env.observation_space.n, env.action_space.n)))

    def action(self, state):
        return self.pi.action(state)

    def train_episode(self):
        # n-step SARSA with importance sampling
        hp: NStepSARSAHyperparameters = self.hyperparameters
        gamma = hp.gamma
        alpha = hp.alpha
        n = hp.n

        # use random policy to generate episode
        bpi = RandomPolicy(self.env.action_space.n)
        
        states = []
        actions = []
        rewards = []

        episode_G = 0.0
        state, _ = self.env.reset()
        done = False

        while not done:
            action = bpi.action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            # store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            episode_G += reward

            state = next_state
            done = terminated or truncated

        # add final state
        states.append(state)

        T = len(rewards)
        if T == 0:
            return episode_G

        # do n-step updates
        Q = self.pi.Q

        for t in range(T):
            tau_end = min(t + n, T)

            # calculate n-step return
            G = 0.0
            for k in range(t, tau_end):
                G += (gamma ** (k - t)) * rewards[k]
            
            # add bootstrap value
            if t + n < T:
                next_state = states[t + n]
                next_action = self.pi.action(next_state)
                G += (gamma ** n) * Q[next_state, next_action]

            # importance sampling ratio
            rho = 1.0
            for k in range(t + 1, tau_end):
                pi_prob = self.pi.action_prob(states[k], actions[k])
                b_prob = bpi.action_prob(states[k], actions[k])
                if pi_prob == 0.0:
                    rho = 0.0
                    break
                rho *= pi_prob / b_prob

            # update Q value
            s_t = states[t]
            a_t = actions[t]
            Q[s_t, a_t] += alpha * rho * (G - Q[s_t, a_t])

        return episode_G
