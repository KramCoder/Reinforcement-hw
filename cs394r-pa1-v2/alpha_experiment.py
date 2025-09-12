#!/usr/bin/env python3
"""
Experiment with different alpha (step-size) values for constant step-size method
"""

import numpy as np
import matplotlib.pyplot as plt
from assignments.bandit import BanditSolver, BanditSolverHyperparameters, ConstantStepSizeMethod
from assignments.bandit_env import BanditEnvironment
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm

def run_alpha_experiment():
    """Run experiments with different alpha values"""
    
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    num_simulations = 100
    num_iterations = 5000
    k = 10
    epsilon = 0.1
    
    results = {}
    
    for alpha in alphas:
        print(f"\nRunning experiment with alpha={alpha}")
        
        rewards = np.zeros(num_iterations)
        optimal_actions = np.zeros(num_iterations)
        
        for sim in tqdm(range(num_simulations)):
            env = TimeLimit(BanditEnvironment(k=k), max_episode_steps=num_iterations)
            hyperparams = BanditSolverHyperparameters(epsilon=epsilon, alpha=alpha)
            solver = BanditSolver(env, hyperparams, ConstantStepSizeMethod)
            r, opt = solver.train_episode()
            rewards += r
            optimal_actions += opt
        
        results[alpha] = {
            'rewards': rewards / num_simulations,
            'optimal': optimal_actions / num_simulations
        }
    
    return results, alphas

def plot_alpha_results(results, alphas):
    """Plot the results of alpha experiments"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    ax = axes[0]
    for alpha in alphas:
        ax.plot(results[alpha]['rewards'], label=f'α={alpha}', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_title('Effect of Step Size (α) on Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot optimal actions
    ax = axes[1]
    for alpha in alphas:
        ax.plot(results[alpha]['optimal'], label=f'α={alpha}', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('% Optimal Action')
    ax.set_title('Effect of Step Size (α) on Optimal Action Selection')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Impact of Step Size on Constant Step-Size Method Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig('alpha_experiment.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as alpha_experiment.png")

def analyze_convergence_speed(results, alphas):
    """Analyze how quickly different alpha values converge"""
    
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    
    for alpha in alphas:
        rewards = results[alpha]['rewards']
        optimal = results[alpha]['optimal']
        
        # Find when performance reaches 80% of final value
        final_reward = np.mean(rewards[-500:])
        target_reward = 0.8 * final_reward
        
        convergence_step = None
        for i in range(100, len(rewards)):
            if np.mean(rewards[i-100:i]) >= target_reward:
                convergence_step = i
                break
        
        print(f"\nAlpha = {alpha}:")
        print(f"  Final avg reward: {final_reward:.3f}")
        print(f"  80% convergence at step: {convergence_step if convergence_step else 'Not reached'}")
        print(f"  Final optimal action rate: {np.mean(optimal[-500:]):.1%}")

if __name__ == "__main__":
    print("Running alpha (step-size) experiments...")
    results, alphas = run_alpha_experiment()
    plot_alpha_results(results, alphas)
    analyze_convergence_speed(results, alphas)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Performance over last 1000 steps")
    print("="*60)
    
    for alpha in alphas:
        avg_reward = np.mean(results[alpha]['rewards'][-1000:])
        avg_optimal = np.mean(results[alpha]['optimal'][-1000:])
        reward_std = np.std(results[alpha]['rewards'][-1000:])
        
        print(f"\nAlpha = {alpha}:")
        print(f"  Average Reward: {avg_reward:.3f} (±{reward_std:.3f})")
        print(f"  Optimal Action: {avg_optimal:.1%}")