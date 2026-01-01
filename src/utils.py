import numpy as np

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t+1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
        
    returns = advantages + values
    return advantages, returns

def normalize(x):
    """
    Normalize advantages for stability.
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

class Buffer:
    """
    Storage for trajectories.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self):
        data = {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones)
        }
        self.clear()
        return data

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
