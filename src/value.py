import numpy as np

class ValueNetwork:
    """
    A simple 2-layer MLP for the value function.
    Outputs a single scalar value.
    """
    def __init__(self, input_dim, hidden_dim, lr=1e-3):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def update(self, states, targets):
        """
        Update value network using MSE loss.
        """
        # Forward pass
        values = self.forward(states)
        
        # Loss = mean((values - targets)^2)
        # dL/dvalues = 2 * (values - targets) / batch_size
        batch_size = states.shape[0]
        dz2 = 2 * (values - targets) / batch_size
        
        # Backprop
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU gradient
        
        dW1 = np.dot(states.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights (Gradient Descent)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        return np.mean((values - targets)**2)
