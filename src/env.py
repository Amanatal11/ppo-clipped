import numpy as np

class SimpleEnv:
    """
    A simple 1D environment where the agent starts at a random position
    and needs to reach a target position (0.0).
    """
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.state_dim = 1
        self.action_dim = 2  # 0: Left (-0.1), 1: Right (+0.1)
        self.reset()

    def reset(self):
        # Random position between -1 and 1
        self.pos = np.random.uniform(-1, 1)
        self.steps = 0
        return np.array([self.pos], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        
        # Apply action
        if action == 0:
            self.pos -= 0.1
        else:
            self.pos += 0.1
            
        # Clip position
        self.pos = np.clip(self.pos, -1.5, 1.5)
        
        # Reward: negative distance to target (0.0)
        reward = -abs(self.pos)
        
        # Done if close to target or max steps reached
        done = abs(self.pos) < 0.05 or self.steps >= self.max_steps
        
        # Bonus reward for reaching target
        if abs(self.pos) < 0.05:
            reward += 10.0
            
        return np.array([self.pos], dtype=np.float32), reward, done, {}
