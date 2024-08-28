import numpy as np
import random
import matplotlib.pyplot as plt

# Grid world dimensions
ROWS, COLS = 3, 3

# Actions (up, down, left, right)
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Reward function
def reward(s, a, next_s):
    if next_s == (2, 2):  # Goal reached
        return 10
    elif next_s == (1, 1):  # Obstacle hit
        return -5
    else:
        return -1  # Step penalty

# Q-learning algorithm
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9):
        self.q_values = np.zeros((ROWS, COLS, len(ACTIONS)))
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, s, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(range(len(ACTIONS)))
        else:
            return np.argmax(self.q_values[s[0], s[1]])

    def update_q_values(self, s, a, next_s, r):
        q_value = self.q_values[s[0], s[1], a]
        if 0 <= next_s[0] < ROWS and 0 <= next_s[1] < COLS:
            next_q_value = np.max(self.q_values[next_s[0], next_s[1]])
        else:
            next_q_value = 0
        self.q_values[s[0], s[1], a] = q_value + self.alpha * (r + self.gamma * next_q_value - q_value)

# Train the agent
agent = QLearningAgent()
s = (0, 0)  # Starting position
episode_rewards = []
for episode in range(1000):
    episode_reward = 0
    s = (0, 0)
    done = False
    while not done:
        a = agent.choose_action(s)
        next_s = (s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1])
        if 0 <= next_s[0] < ROWS and 0 <= next_s[1] < COLS:
            r = reward(s, a, next_s)
            episode_reward += r
            agent.update_q_values(s, a, next_s, r)
            s = next_s
            if next_s == (2, 2):  # Goal reached
                done = True
        else:
            r = -10  # Penalty for moving outside the grid world
            episode_reward += r
            agent.update_q_values(s, a, s, r)
    episode_rewards.append(episode_reward)

# Print the optimal policy
print("Optimal policy:")
for row in range(ROWS):
    for col in range(COLS):
        a = np.argmax(agent.q_values[row, col])
        print(["↑", "↓", "←", "→"][a], end=" ")
    print()

# Plot the episode rewards
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-Learning Rewards")
plt.show()

# Plot the Q-values
plt.figure(figsize=(10, 8))
for a in range(len(ACTIONS)):
    plt.subplot(2, 2, a + 1)
    plt.imshow(agent.q_values[:, :, a], cmap="hot", interpolation="nearest")
    plt.title(["Up", "Down", "Left", "Right"][a])
    plt.colorbar()
plt.tight_layout()
plt.show()