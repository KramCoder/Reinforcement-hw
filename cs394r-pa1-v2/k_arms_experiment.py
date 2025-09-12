#!/usr/bin/env python3
"""
Experiment with different numbers of arms (k) to understand scaling effects
"""

import numpy as np
import matplotlib.pyplot as plt
from assignments.bandit import BanditSolver, BanditSolverHyperparameters, SampleAverageMethod, ConstantStepSizeMethod
from assignments.bandit_env import BanditEnvironment
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm

def run_k_arms_experiment():
    """Run experiments with different numbers of arms"""
    
    k_values = [2, 5, 10, 20, 50, 100]
    num_simulations = 50  # Fewer simulations for larger k values
    num_iterations = 3000
    epsilon = 0.1
    alpha = 0.1
    
    results = {}
    
    for k in k_values:
        print(f"\nRunning experiment with k={k} arms")
        
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
        
        results[k] = {
            'sample_avg_rewards': sample_avg_rewards / num_simulations,
            'sample_avg_optimal': sample_avg_optimal / num_simulations,
            'constant_rewards': constant_rewards / num_simulations,
            'constant_optimal': constant_optimal / num_simulations
        }
    
    return results, k_values

def plot_k_arms_results(results, k_values):
    """Plot the results of k-arms experiments"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards for different k values - Sample Average
    ax = axes[0, 0]
    for k in k_values:
        ax.plot(results[k]['sample_avg_rewards'], label=f'k={k}', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_title('Sample Average Method - Effect of Number of Arms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot optimal actions for different k values - Sample Average
    ax = axes[0, 1]
    for k in k_values:
        ax.plot(results[k]['sample_avg_optimal'], label=f'k={k}', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('% Optimal Action')
    ax.set_title('Sample Average Method - Optimal Action Selection')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot rewards for different k values - Constant Step Size
    ax = axes[1, 0]
    for k in k_values:
        ax.plot(results[k]['constant_rewards'], label=f'k={k}', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_title('Constant Step Size Method - Effect of Number of Arms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot optimal actions for different k values - Constant Step Size
    ax = axes[1, 1]
    for k in k_values:
        ax.plot(results[k]['constant_optimal'], label=f'k={k}', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('% Optimal Action')
    ax.set_title('Constant Step Size Method - Optimal Action Selection')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Impact of Number of Arms on Bandit Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig('k_arms_experiment.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as k_arms_experiment.png")

def plot_scaling_analysis(results, k_values):
    """Analyze how performance scales with number of arms"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract final performance metrics
    final_rewards_sa = []
    final_optimal_sa = []
    final_rewards_cs = []
    final_optimal_cs = []
    
    for k in k_values:
        final_rewards_sa.append(np.mean(results[k]['sample_avg_rewards'][-500:]))
        final_optimal_sa.append(np.mean(results[k]['sample_avg_optimal'][-500:]))
        final_rewards_cs.append(np.mean(results[k]['constant_rewards'][-500:]))
        final_optimal_cs.append(np.mean(results[k]['constant_optimal'][-500:]))
    
    # Plot scaling of rewards
    ax = axes[0]
    ax.plot(k_values, final_rewards_sa, 'o-', label='Sample Average', markersize=8)
    ax.plot(k_values, final_rewards_cs, 's-', label='Constant Step Size', markersize=8)
    ax.set_xlabel('Number of Arms (k)')
    ax.set_ylabel('Final Average Reward')
    ax.set_title('Reward Scaling with Number of Arms')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot scaling of optimal action selection
    ax = axes[1]
    ax.plot(k_values, final_optimal_sa, 'o-', label='Sample Average', markersize=8)
    ax.plot(k_values, final_optimal_cs, 's-', label='Constant Step Size', markersize=8)
    ax.set_xlabel('Number of Arms (k)')
    ax.set_ylabel('Final % Optimal Action')
    ax.set_title('Optimal Action Selection Scaling')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Scaling Analysis: Performance vs Number of Arms', fontsize=14)
    plt.tight_layout()
    plt.savefig('k_arms_scaling.png', dpi=150, bbox_inches='tight')
    print("Scaling plot saved as k_arms_scaling.png")

if __name__ == "__main__":
    print("Running k-arms experiments...")
    results, k_values = run_k_arms_experiment()
    plot_k_arms_results(results, k_values)
    plot_scaling_analysis(results, k_values)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Final Performance (last 500 steps)")
    print("="*60)
    
    print("\n{:<10} {:<25} {:<25}".format("K Arms", "Sample Average", "Constant Step Size"))
    print("-"*60)
    
    for k in k_values:
        sa_reward = np.mean(results[k]['sample_avg_rewards'][-500:])
        sa_optimal = np.mean(results[k]['sample_avg_optimal'][-500:])
        cs_reward = np.mean(results[k]['constant_rewards'][-500:])
        cs_optimal = np.mean(results[k]['constant_optimal'][-500:])
        
        print("{:<10} R:{:.3f} Opt:{:.1%}      R:{:.3f} Opt:{:.1%}".format(
            k, sa_reward, sa_optimal, cs_reward, cs_optimal))