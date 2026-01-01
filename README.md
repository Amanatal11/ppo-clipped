# PPO from Scratch (NumPy Only)

A clean, educational implementation of **Proximal Policy Optimization (PPO)** using **only Python and NumPy**.

This project demonstrates how to build a state-of-the-art Reinforcement Learning algorithm without relying on high-level deep learning frameworks like PyTorch or TensorFlow. It features manual backpropagation, a custom environment, and the core PPO clipped surrogate objective.

## 🚀 Key Features

*   **No Deep Learning Libraries**: All neural networks (Policy and Value) are implemented from scratch using NumPy.
*   **Manual Backpropagation**: Gradients are computed manually for both the Policy (Softmax) and Value (MSE) networks.
*   **PPO Clipped Objective**: Implements the trust-region update rule: $L^{CLIP}(\theta) = \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$.
*   **Generalized Advantage Estimation (GAE)**: Uses GAE for stable advantage computation.
*   **Custom Environment**: Includes a simple 1D "Target Reach" environment for verification.

## 📂 Project Structure

```text
src/
├── env.py       # Simple 1D custom environment
├── policy.py    # Policy network with manual backprop & PPO clipping
├── value.py     # Value network with manual backprop & MSE loss
├── ppo.py       # PPO update logic
├── utils.py     # GAE and trajectory buffer
└── train.py     # Main training loop
```

## 🛠️ Installation & Usage

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/yourusername/ppo-clipped.git
    cd ppo-clipped
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install numpy
    ```

4.  **Run the training script**:
    ```bash
    python src/train.py
    ```

## 📊 Results

The agent learns to reach the target position (0.0) in the 1D environment. You should see the average reward improve from ~ -20 to ~ +7 over 500 episodes.

```text
Episode 20 | Avg Reward: -19.03
...
Episode 500 | Avg Reward: 6.44
```

## 🧠 Educational Value

This codebase is designed to be read. Check `src/policy.py` to see how the PPO loss gradient is manually derived and applied to the network weights, and `src/utils.py` to understand how GAE balances bias and variance in advantage estimation.
