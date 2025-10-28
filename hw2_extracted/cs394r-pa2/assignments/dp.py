import gymnasium as gym
from typing import Tuple

import numpy as np
from interfaces.policy import Policy

from assignments.policy_deterministic_greedy import Policy_DeterministicGreedy

def value_prediction(
    env: gym.Env, 
    pi: Policy,
    initV: np.array,
    theta: float,
    gamma: float
) -> Tuple[np.array, np.array]:
    """
    Runs the value prediction algorithm to estimate the value function for a given policy.

    Sutton & Barto, p. 75, "Value Prediction"
    
    Parameters:
        env (gym.Env): environment with model information, i.e. you know transition dynamics and reward function
        pi (Policy): The policy to evaluate (behavior policy)
        initV (np.ndarray): Initial V(s); numpy array shape of [nS,]
        theta (float): The exit criteria
    Returns:
        tuple: A tuple containing:
            - V (np.ndarray): V_pi function; numpy array shape of [nS]
            - Q (np.ndarray): Q_pi function; numpy array shape of [nS,nA]
    """
    # Transition dynamics and spaces
    P = env.P
    states = env.observation_space.n
    actions = env.action_space.n

    # Initialize V and Q
    V = initV.copy()
    Q = np.zeros((states, actions))

    # Iterative policy evaluation (synchronous sweep until convergence)
    while True:
        delta = 0.0
        # Update Q from current V for all state-action pairs
        for s in range(states):
            for a in range(actions):
                q_sa = 0.0
                for prob, next_state, reward, done in P[s][a]:
                    q_sa += prob * (reward + gamma * (0.0 if done else V[next_state]))
                Q[s, a] = q_sa

        # Update V using policy action probabilities over Q
        for s in range(states):
            v_old = V[s]
            v_new = 0.0
            for a in range(actions):
                v_new += pi.action_prob(s, a) * Q[s, a]
            V[s] = v_new
            delta = max(delta, abs(v_old - V[s]))

        if delta < theta:
            break

    return V, Q

def value_iteration(env: gym.Env, initV: np.ndarray, theta: float, gamma: float) -> Tuple[np.array, Policy]:
    """
    Parameters:
        env (EnvWithModel): environment with model information, i.e. you know transition dynamics and reward function
        initV (np.ndarray): initial V(s); numpy array shape of [nS,]
        theta (float): exit criteria

    Returns:
        tuple: A tuple containing:
            - value (np.ndarray): optimal value function; shape of [nS]
            - policy (GreedyQPolicy): optimal deterministic policy
    """

    # Setup
    nS: int = env.observation_space.n
    nA: int = env.action_space.n
    V: np.ndarray = initV.copy()
    Q: np.ndarray = np.zeros((nS, nA))
    P: np.ndarray = env.P

    # Value Iteration: repeatedly update V using Bellman optimality backup
    while True:
        delta = 0.0
        for s in range(nS):
            v_old = V[s]

            # Compute Q(s,a) for all actions under current V
            for a in range(nA):
                q_sa = 0.0
                for prob, next_state, reward, done in P[s][a]:
                    q_sa += prob * (reward + gamma * (0.0 if done else V[next_state]))
                Q[s, a] = q_sa

            # Greedy update for V
            V[s] = np.max(Q[s])
            delta = max(delta, abs(v_old - V[s]))

        if delta < theta:
            break

    # Derive greedy policy from optimal Q
    pi: Policy_DeterministicGreedy = Policy_DeterministicGreedy(Q)
    return V, pi
