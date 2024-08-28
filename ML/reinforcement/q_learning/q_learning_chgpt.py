import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import patches

# Parameters
gamma = 0.9  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Exploration factor
episodes = 1000
grid_size = 4

# Initialize Q-table (states x actions)
Q_table = np.zeros((grid_size * grid_size, 4))  # 4 actions: up, down, left, right

# Environment setup
def get_next_state(state, action):
    x, y = divmod(state, grid_size)
    
    if action == 0 and x > 0:  # Up
        x -= 1
    elif action == 1 and x < grid_size - 1:  # Down
        x += 1
    elif action == 2 and y > 0:  # Left
        y -= 1
    elif action == 3 and y < grid_size - 1:  # Right
        y += 1
    
    return x * grid_size + y

def get_reward(state):
    if state == grid_size * grid_size - 1:
        return 1  # Goal reward
    return -0.01  # Small penalty for each step

# Q-Learning algorithm
for episode in range(episodes):
    state = random.randint(0, grid_size * grid_size - 1)  # Start at random position
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)  # Explore: random action
        else:
            action = np.argmax(Q_table[state])  # Exploit: choose best action from Q-table
        
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        
        # Update Q-value
        best_future_q = np.max(Q_table[next_state])
        Q_table[state, action] += alpha * (reward + gamma * best_future_q - Q_table[state, action])
        
        state = next_state
        
        if state == grid_size * grid_size - 1:
            done = True  # Goal reached

# Plot the learned policy
def plot_policy(Q_table, grid_size):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    
    for state in range(grid_size * grid_size):
        x, y = divmod(state, grid_size)
        
        action = np.argmax(Q_table[state])
        if action == 0:  # Up
            ax.arrow(y + 0.5, grid_size - x - 0.5, 0, 0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        elif action == 1:  # Down
            ax.arrow(y + 0.5, grid_size - x - 0.5, 0, -0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        elif action == 2:  # Left
            ax.arrow(y + 0.5, grid_size - x - 0.5, -0.3, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        elif action == 3:  # Right
            ax.arrow(y + 0.5, grid_size - x - 0.5, 0.3, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    # Draw goal state
    goal = patches.Rectangle((grid_size - 1, 0), 1, 1, facecolor="green")
    ax.add_patch(goal)
    
    # Draw grid
    for i in range(grid_size + 1):
        ax.plot([0, grid_size], [i, i], color='black')
        ax.plot([i, i], [0, grid_size], color='black')
    
    plt.title("Learned Policy with Q-Learning")
    plt.gca().invert_yaxis()
    plt.show()

# Plot the learned policy
plot_policy(Q_table, grid_size)
