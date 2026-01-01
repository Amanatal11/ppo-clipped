import numpy as np
from env import SimpleEnv
from policy import PolicyNetwork
from value import ValueNetwork
from ppo import PPO
from utils import Buffer

def train():
    # Hyperparameters
    state_dim = 1
    action_dim = 2
    hidden_dim = 32
    lr_policy = 1e-3
    lr_value = 2e-3
    gamma = 0.99
    lam = 0.95
    epsilon = 0.2
    epochs = 10
    batch_size = 64
    max_episodes = 500
    update_timestep = 200 # Update every 200 steps
    
    # Initialize components
    env = SimpleEnv()
    policy = PolicyNetwork(state_dim, hidden_dim, action_dim, lr=lr_policy)
    value = ValueNetwork(state_dim, hidden_dim, lr=lr_value)
    ppo = PPO(policy, value, gamma, lam, epsilon, epochs, batch_size)
    buffer = Buffer()
    
    timestep = 0
    episode_rewards = []
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(env.max_steps):
            timestep += 1
            
            # Sample action
            action, log_prob = policy.sample_action(state)
            val = value.forward(state.reshape(1, -1))[0, 0]
            
            # Step environment
            next_state, reward, done, _ = env.step(action)
            
            # Store in buffer
            buffer.store(state, action, reward, val, log_prob, done)
            
            state = next_state
            episode_reward += reward
            
            # Update PPO
            if timestep % update_timestep == 0:
                data = buffer.get()
                # Get next value for GAE
                next_val = value.forward(next_state.reshape(1, -1))[0, 0] if not done else 0
                
                ppo.perform_update(
                    data['states'],
                    data['actions'],
                    data['rewards'],
                    data['values'],
                    data['log_probs'],
                    data['dones'],
                    next_val
                )
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1} | Avg Reward: {avg_reward:.2f}")
            
    return episode_rewards

if __name__ == "__main__":
    train()
