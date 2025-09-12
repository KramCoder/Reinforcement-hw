#!/usr/bin/env python3
"""
Experiment with different epsilon values to understand exploration-exploitation tradeoff
"""

import numpy as np
import matplotlib.pyplot as plt
from assignments.bandit import BanditSolver, BanditSolverHyperparameters, SampleAverageMethod, ConstantStepSizeMethod
from assignments.bandit_env import BanditEnvironment
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm

def run_epsilon_experiment():
    """Run experiments with different epsilon values"""
    
    epsilons = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
    num_simulations = 100
    num_iterations = 5000
    k = 10
    alpha = 0.1
    
    results = {}
    
    for epsilon in epsilons:
        print(f"\nRunning experiment with epsilon={epsilon}")
        
        # Arrays to store results
        sample_avg_rewards = np.zeros(num_iterations)
        sample_avg_optimal = np.zeros(num_iterations)
        constant_rewards = np.zeros(num_iterations)
        constant_optimal = np.zeros(num_iterations)
        
        for sim in tqdm(range(num_simulations)):
            # Sample Average Method
            env = TimeLimit(BanditEnvironment(k=k), max_episode_steps=num_iterations)
            hyperparams = BanditSolverHyperparameters(epsilon=epsilon)
            solver = BanditSolver(env, hyperparams, SampleAverageMethod)
            rewards, optimal_actions = solver.train_episode()
            sample_avg_rewards += rewards
            sample_avg_optimal += optimal_actions
            
            # Constant Step Size Method
            env = TimeLimit(BanditEnvironment(k=k), max_episode_steps=num_iterations)
            hyperparams = BanditSolverHyperparameters(epsilon=epsilon, alpha=alpha)
            solver = BanditSolver(env, hyperparams, ConstantStepSizeMethod)
            rewards, optimal_actions = solver.train_episode()
            constant_rewards += rewards
            constant_optimal += optimal_actions
        
        # Average over simulations
        results[epsilon] = {
            'sample_avg_rewards': sample_avg_rewards / num_simulations,
            'sample_avg_optimal': sample_avg_optimal / num_simulations,
            'constant_rewards': constant_rewards / num_simulations,
            'constant_optimal': constant_optimal / num_simulations
        }
    
    return results, epsilons

def plot_epsilon_results(results, epsilons):
    """Plot the results of epsilon experiments"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards for sample average
    ax = axes[0, 0]
    for epsilon in epsilons:
        ax.plot(results[epsilon]['sample_avg_rewards'], 
                label=f'ε={epsilon}', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_title('Sample Average Method - Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot optimal actions for sample average
    ax = axes[0, 1]
    for epsilon in epsilons:
        ax.plot(results[epsilon]['sample_avg_optimal'], 
                label=f'ε={epsilon}', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('% Optimal Action')
    ax.set_title('Sample Average Method - Optimal Actions')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot rewards for constant step size
    ax = axes[1, 0]
    for epsilon in epsilons:
        ax.plot(results[epsilon]['constant_rewards'], 
                label=f'ε={epsilon}', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_title('Constant Step Size (α=0.1) - Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot optimal actions for constant step size
    ax = axes[1, 1]
    for epsilon in epsilons:
        ax.plot(results[epsilon]['constant_optimal'], 
                label=f'ε={epsilon}', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('% Optimal Action')
    ax.set_title('Constant Step Size (α=0.1) - Optimal Actions')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Epsilon on Bandit Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig('epsilon_experiment.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as epsilon_experiment.png")

if __name__ == "__main__":
    print("Running epsilon experiments...")
    results, epsilons = run_epsilon_experiment()
    plot_epsilon_results(results, epsilons)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY: Average performance over last 1000 steps")
    print("="*60)
    
    for epsilon in epsilons:
        avg_reward_sa = np.mean(results[epsilon]['sample_avg_rewards'][-1000:])
        avg_optimal_sa = np.mean(results[epsilon]['sample_avg_optimal'][-1000:])
        avg_reward_cs = np.mean(results[epsilon]['constant_rewards'][-1000:])
        avg_optimal_cs = np.mean(results[epsilon]['constant_optimal'][-1000:])
        
        print(f"\nEpsilon = {epsilon}:")
        print(f"  Sample Average: Reward={avg_reward_sa:.3f}, Optimal={avg_optimal_sa:.1%}")
        print(f"  Constant Step:  Reward={avg_reward_cs:.3f}, Optimal={avg_optimal_cs:.1%}")