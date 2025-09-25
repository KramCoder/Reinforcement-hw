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
    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    # Hint: To get the action probability, use pi.action_prob(state,action)
    # Hint: Use the "env.P" to get the transition probabilities.
    #    env.P[state][action] returns a list of tuples [(prob, next_state, reward, done)]
    #    (Both our custom environments and OpenAI Gym environments have this attribute)
    #####################
    P = env.P
    nS = env.observation_space.n
    nA = env.action_space.n

    V = initV.copy()

    while True:
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            v_new = 0.0
            for a in range(nA):
                pi_sa = pi.action_prob(s, a)
                if pi_sa == 0:
                    continue
                for prob, s_next, reward, done in P[s][a]:
                    v_new += pi_sa * prob * (reward + gamma * V[s_next])
            delta = max(delta, abs(v_old - v_new))
            V[s] = v_new
        if delta < theta:
            break

    # Compute Q(s,a)
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            q_sa = 0.0
            for prob, s_next, reward, done in P[s][a]:
                q_sa += prob * (reward + gamma * V[s_next])
            Q[s, a] = q_sa

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

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    # Hint: Use the "env.P" to get the transition probabilities.
    #    env.P[state][action] returns a list of tuples [(prob, next_state, reward, done)]
    #    (Both our custom environments and OpenAI Gym environments have this attribute)
    # Hint: Try updating the Q function in the `pi` policy object
    #####################
    nS = env.observation_space.n
    nA = env.action_space.n

    V = initV.copy()
    P = env.P

    while True:
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            q_values = []
            for a in range(nA):
                q_sa = 0.0
                for prob, s_next, reward, done in P[s][a]:
                    q_sa += prob * (reward + gamma * V[s_next])
                q_values.append(q_sa)
            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break

    # Compute Q and create greedy policy
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            q_sa = 0.0
            for prob, s_next, reward, done in P[s][a]:
                q_sa += prob * (reward + gamma * V[s_next])
            Q[s, a] = q_sa

    pi = Policy_DeterministicGreedy(Q)

    return V, pi
