# CS 394R PA2 - Dynamic Programming Solution

## Files Implemented

### Required Deliverables
- `assignments/dp.py` - Contains value prediction and value iteration algorithms
- `assignments/policy_deterministic_greedy.py` - Contains deterministic greedy policy class

## What Was Implemented

### 1. Policy_DeterministicGreedy Class
- **`action(state)`**: Returns action with highest Q-value using `np.argmax()`
- **`action_prob(state, action)`**: Returns 1.0 for optimal action, 0.0 for others

### 2. Value Prediction Algorithm
- Implements Iterative Policy Evaluation (Sutton & Barto p.75)
- Evaluates given policy π to compute V^π(s) and Q^π(s,a)
- Converges using theta threshold
- Works with both optimal and random policies

### 3. Value Iteration Algorithm  
- Implements Value Iteration (Sutton & Barto p.83)
- Finds optimal value function V*(s) and optimal policy π*
- Updates V by taking max over Q-values
- Returns optimal greedy policy

## Testing

### Quick Test
```bash
python test.py dp
```

### Visualization (if you have display)
```bash
# Simple environment
python run.py dp --environment OneStateMDP-v0

# Grid world
python run.py dp --environment GridWorld2x2-v0

# Frozen lake
python run.py dp --environment WrappedFrozenLake-v0
```

## Test Results
✅ **All 15 tests passing**
- OneStateMDP: V, π tests
- GridWorld2x2: V, π, Q tests  
- Value prediction with optimal and random policies
- Complex environments: FrozenLake, Taxi

## Key Implementation Details

### Value Prediction
```python
# Iterative Policy Evaluation
while True:
    delta = 0
    for s in range(states):
        v = 0
        for a in range(actions):
            action_prob = pi.action_prob(s, a)
            for prob, next_state, reward, done in P[s][a]:
                v += action_prob * prob * (reward + gamma * V_old[next_state])
        V[s] = v
        delta = max(delta, abs(V[s] - V_old[s]))
    if delta < theta:
        break
```

### Value Iteration
```python
# Value Iteration
while True:
    delta = 0
    for s in range(nS):
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in P[s][a]:
                q_values[a] += prob * (reward + gamma * V_old[next_state])
        V[s] = np.max(q_values)
        Q[s, :] = q_values
        delta = max(delta, abs(V[s] - V_old[s]))
    if delta < theta:
        break
```

## Installation
```bash
pip install -r requirements.txt
```

## Files Available for Download
- `cs394r-pa2-solution.zip` - Just the two required deliverable files
- `cs394r-pa2-complete-solution.zip` - Full package with testing files