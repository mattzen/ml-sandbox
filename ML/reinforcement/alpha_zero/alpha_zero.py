import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
# ...
# This simplified version should run much faster while still demonstrating some key 
# concepts inspired by AlphaZero:

# It uses a Monte Carlo approach for learning, updating Q-values based on game outcomes.
# The agents balance exploration and exploitation using an epsilon-greedy strategy.
# The learning progress is visualized over time, similar to how you might track an AlphaZero agent's improvement.

# Key differences from a full AlphaZero implementation:

# No neural network: We use a simple Q-value table instead.
# No tree search: Actions are chosen directly from Q-values.
# Simpler game: Tic-Tac-Toe instead of more complex games like Chess or Go.
# This script should run much faster than the previous version.
# It will train two agents playing against each other for 1000 episodes and
# then display a graph showing the learning progress of the first agent.
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def get_valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_move(self, move):
        i, j = move
        if self.board[i, j] == 0:
            self.board[i, j] = self.current_player
            self.current_player *= -1
            return True
        return False

    def check_winner(self):
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3 or abs(np.sum(self.board[:, i])) == 3:
                return self.board[i, 0]
        if abs(np.sum(np.diag(self.board))) == 3 or abs(np.sum(np.diag(np.fliplr(self.board)))) == 3:
            return self.board[1, 1]
        if len(self.get_valid_moves()) == 0:
            return 0  # Draw
        return None  # Game not finished

class MonteCarloAgent:
    def __init__(self, epsilon=0.1, alpha=0.5):
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.epsilon = epsilon
        self.alpha = alpha

    def choose_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            state_key = self.state_to_key(state)
            return max(valid_moves, key=lambda move: self.q_values[state_key][move])

    def update(self, state, action, reward):
        state_key = self.state_to_key(state)
        old_value = self.q_values[state_key][action]
        self.q_values[state_key][action] += self.alpha * (reward - old_value)

    @staticmethod
    def state_to_key(state):
        return tuple(map(tuple, state))

def play_game(agent1, agent2):
    game = TicTacToe()
    states_actions = []

    while True:
        current_agent = agent1 if game.current_player == 1 else agent2
        valid_moves = game.get_valid_moves()
        action = current_agent.choose_action(game.board, valid_moves)
        states_actions.append((game.board.copy(), action))
        game.make_move(action)

        winner = game.check_winner()
        if winner is not None:
            return winner, states_actions

def train_agents(num_episodes=1000):
    agent1 = MonteCarloAgent()
    agent2 = MonteCarloAgent()
    win_rates = []

    for episode in range(num_episodes):
        winner, states_actions = play_game(agent1, agent2)
        
        if winner == 1:
            win_rates.append(1)
            reward = 1
        elif winner == -1:
            win_rates.append(0)
            reward = -1
        else:
            win_rates.append(0.5)
            reward = 0

        for state, action in states_actions:
            if state[action[0], action[1]] == 1:
                agent1.update(state, action, reward)
            else:
                agent2.update(state, action, -reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Win rate: {np.mean(win_rates[-100:]):.2f}")

    return win_rates

# Run training
win_rates = train_agents(num_episodes=1000)

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(range(len(win_rates)), win_rates)
plt.title('AlphaZero-inspired Agent Learning Progress')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.ylim(0, 1)
plt.show()