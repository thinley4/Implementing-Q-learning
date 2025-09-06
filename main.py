import numpy as np
import gymnasium as gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v1", render_mode="ansi")

# creating q table

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

print("Initial Q-table")
print(q_table)

# Initializing Q-learning parameters

num_episodes = 10000  # total number of episodes we want the agent to play during training

max_steps_per_episode = 100  # maximum number of steps that our agent is allowed to
# take within a single episode. So, if by the one-hundredth step,
# the agent hasn't reached the frisbee or fallen through a hole, then the
# episode will terminate with the agent receiving zero points.

learning_rate = (
    0.1  # how quickly the agent abandons the previous Q-value in the Q-table
)
discount_rate = (
    0.99  # how much the agent prioritizes future rewards over immediate rewards
)

exploration_rate = 1
# The max and min are just bounds to how large or small our exploration rate can be
max_exploration_rate = 1
min_exploration_rate = 0.01

# rate at which the exploration_rate will decay.
exploration_decay_rate = 0.001

# Implement the actual Q-learning algorithm

# Coding the Q-learning algorithm training loop

rewards_all_episodes = []  # hold all of the rewards we'll get from each episode

# Q-learning algorithm
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()[0]
    done = False  # done variable just keeps track of whether or not our episode is finished
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            # exploitation
            # choose the action that has the highest Q-value in the Q-table
            action = np.argmax(q_table[state, :])
        else:
            # exploration
            action = env.action_space.sample()

        # Take new action
        new_state, reward, done, truncated, info = env.step(action)
        # Update Q-table
        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (
            1 - learning_rate
        ) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        # Set new state
        state = new_state
        # Add new reward
        rewards_current_episode += reward

        if done == True:
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + (
        max_exploration_rate - min_exploration_rate
    ) * np.exp(-exploration_decay_rate * episode)

    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)


# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000


# Print updated Q-table
print("\n\n********Updated-Q-table********\n")
print(q_table)