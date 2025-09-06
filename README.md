# Frozen Lake Q-Learning (Implementing Q-Learning in code with Python)

A simple Q-learning implementation to solve the FrozenLake environment from OpenAI Gymnasium.

## What it does

The agent learns to navigate a frozen lake to reach a goal while avoiding holes using Q-learning reinforcement learning algorithm.

## Requirements

```bash
pip install numpy gymnasium
```

## How to run

```bash
python main.py
```

## Parameters

- **Episodes**: 10,000 training episodes
- **Learning rate**: 0.1
- **Discount rate**: 0.99
- **Exploration rate**: Decays from 1.0 to 0.01

## Output

- Initial Q-table (all zeros)
![initial_q_table](https://github.com/thinley4/Implementing-Q-learning/blob/main/output/initial_q_table.png)

- Average reward per 1000 episodes during training
![reward](https://github.com/thinley4/Implementing-Q-learning/blob/main/output/avg_reward.png)

- Final trained Q-table
![final_q_table](https://github.com/thinley4/Implementing-Q-learning/blob/main/output/final_q_table.png)

The agent starts with random exploration and gradually learns optimal actions through trial and error.

## Credits

Code adapted from: [DeepLizard Q-Learning Tutorial](https://deeplizard.com/learn/video/HGeI30uATws)