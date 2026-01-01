import numpy as np

class PolicyNetwork:
    """
    A simple 2-layer MLP for the policy network.
    Outputs action probabilities using Softmax.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, lr=3e-4):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.probs = self.softmax(self.z2)
        return self.probs

    def sample_action(self, state):
        state = state.reshape(1, -1)
        probs = self.forward(state)[0]
        action = np.random.choice(len(probs), p=probs)
        return action, np.log(probs[action] + 1e-8)

    def update(self, states, actions, advantages, old_log_probs, epsilon=0.2):
        """
        PPO Clipped Surrogate Objective update.
        """
        # Forward pass
        probs = self.forward(states)
        
        # Get log probs of taken actions
        batch_size = states.shape[0]
        action_probs = probs[np.arange(batch_size), actions]
        log_probs = np.log(action_probs + 1e-8)
        
        # Ratio r_t(theta)
        ratios = np.exp(log_probs - old_log_probs)
        
        # Surrogate objectives
        surr1 = ratios * advantages
        surr2 = np.clip(ratios, 1 - epsilon, 1 + epsilon) * advantages
        
        # Loss is negative because we want to maximize it
        # We also add an entropy bonus for exploration
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        # loss = -np.mean(np.minimum(surr1, surr2)) - 0.01 * np.mean(entropy)
        
        # Gradient of the clipped objective w.r.t. log_probs
        # dL/dlog_p = advantages if ratios are not clipped, else 0
        # However, for simplicity and stability, we can use the gradient of the surrogate
        mask = np.where(surr1 <= surr2, 1.0, 0.0)
        # dL/dprobs = (mask * advantages / action_probs) if ratios not clipped
        
        # Backprop through Softmax and MLP
        # dL/dz2 (gradient of cross-entropy-like objective)
        # For PPO, we want to move log_probs in direction of advantages
        # d(log_p)/dz2 = (1 - p) for the chosen action, -p for others
        
        dz2 = np.zeros_like(probs)
        for i in range(batch_size):
            if surr1[i] <= surr2[i] or (ratios[i] > 1+epsilon and advantages[i] < 0) or (ratios[i] < 1-epsilon and advantages[i] > 0):
                # Gradient of ratios * advantages w.r.t z2
                # d(exp(log_p - log_p_old) * A) / dz2 = ratio * A * d(log_p)/dz2
                grad_log_p = -probs[i]
                grad_log_p[actions[i]] += 1
                dz2[i] = ratios[i] * advantages[i] * grad_log_p
        
        # Maximize loss -> Gradient ascent
        dW2 = np.dot(self.a1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU gradient
        
        dW1 = np.dot(states.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
        
        # Update weights
        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1
        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2
