import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log
from typing import List, Tuple

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0

def ucb1(node, parent_visits):
    if node.visits == 0:
        return float('inf')
    return node.value / node.visits + sqrt(2 * log(parent_visits) / node.visits)

def select(node):
    while node.children:
        node = max(node.children, key=lambda c: ucb1(c, node.visits))
    return node

def expand(node):
    if node.state.is_terminal():
        return node
    for action in node.state.get_actions():
        new_state = node.state.take_action(action)
        new_node = Node(new_state, parent=node)
        node.children.append(new_node)
    return np.random.choice(node.children)

def simulate(node):
    state = node.state
    while not state.is_terminal():
        action = np.random.choice(state.get_actions())
        state = state.take_action(action)
    return state.get_reward()

def backpropagate(node, reward):
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent

def mcts(root, num_simulations):
    for _ in range(num_simulations):
        leaf = select(root)
        child = expand(leaf)
        reward = simulate(child)
        backpropagate(child, reward)
    return max(root.children, key=lambda c: c.visits)

class SimpleGame:
    def __init__(self, depth, branching_factor):
        self.depth = depth
        self.branching_factor = branching_factor
        self.current_depth = 0

    def is_terminal(self):
        return self.current_depth == self.depth

    def get_actions(self):
        return list(range(self.branching_factor))

    def take_action(self, action):
        new_game = SimpleGame(self.depth, self.branching_factor)
        new_game.current_depth = self.current_depth + 1
        return new_game

    def get_reward(self):
        return np.random.randint(0, 2)  # Random reward of 0 or 1

def visualize_tree(node, ax, x=0, y=0, dx=1, dy=1):
    if not node.children:
        return

    for i, child in enumerate(node.children):
        nx = x + dx * (i - (len(node.children) - 1) / 2)
        ny = y - dy
        ax.plot([x, nx], [y, ny], 'k-')
        ax.annotate(f'{child.visits}', (nx, ny), xytext=(0, -5), 
                    textcoords='offset points', ha='center', va='top')
        visualize_tree(child, ax, nx, ny, dx/2, dy)

def main():
    game = SimpleGame(depth=4, branching_factor=3)
    root = Node(game)
    best_child = mcts(root, num_simulations=1000)

    fig, ax = plt.subplots(figsize=(12, 8))
    visualize_tree(root, ax)
    ax.set_axis_off()
    plt.title("Monte Carlo Tree Search Visualization")
    plt.tight_layout()
    plt.show()

    print(f"Best action: {root.children.index(best_child)}")
    print(f"Best child visits: {best_child.visits}")

if __name__ == "__main__":
    main()