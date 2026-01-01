import numpy as np
from utils import compute_gae, normalize

class PPO:
    """
    PPO Clipped algorithm implementation.
    """
    def __init__(self, policy, value, gamma=0.99, lam=0.95, epsilon=0.2, epochs=10, batch_size=64):
        self.policy = policy
        self.value = value
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size

    def update(self, buffer):
        data = buffer.get()
        states = data['states']
        actions = data['actions']
        rewards = data['rewards']
        values = data['values']
        log_probs = data['log_probs']
        dones = data['dones']

        # Compute next value for GAE
        next_state = states[-1] # This is a simplification; in a real loop we'd pass the actual next state
        # For our simple env, we can just use 0 if done, or re-run value network
        # But compute_gae handles this if we pass the last value correctly.
        # Let's assume the buffer contains a full trajectory or a fixed window.
        
        # We need the value of the state AFTER the last one in the buffer
        # For simplicity, let's assume the last state in buffer is the terminal state or we have its next value
        # In train.py, we will handle this by passing the next_value explicitly.
        return states, actions, rewards, values, log_probs, dones

    def perform_update(self, states, actions, rewards, values, log_probs, dones, next_value):
        # Compute advantages and returns
        advantages, returns = compute_gae(rewards, values, next_value, dones, self.gamma, self.lam)
        advantages = normalize(advantages)
        
        # Multiple epochs of updates
        batch_size = states.shape[0]
        indices = np.arange(batch_size)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_advantages = advantages[batch_idx]
                b_log_probs = log_probs[batch_idx]
                b_returns = returns[batch_idx].reshape(-1, 1)
                
                # Update policy
                self.policy.update(b_states, b_actions, b_advantages, b_log_probs, self.epsilon)
                
                # Update value
                self.value.update(b_states, b_returns)
