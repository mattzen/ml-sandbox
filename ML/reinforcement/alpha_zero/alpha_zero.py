import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import random

# Tic-Tac-Toe environment
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        return self.board.copy()

    def step(self, action):
        row, col = action // 3, action % 3
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.current_player *= -1
            return self.board.copy(), self.check_winner(), self.is_full()
        return self.board.copy(), None, False

    def check_winner(self):
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3 or abs(np.sum(self.board[:, i])) == 3:
                return self.board[i, 0]
        if abs(np.sum(np.diag(self.board))) == 3 or abs(np.sum(np.diag(np.fliplr(self.board)))) == 3:
            return self.board[1, 1]
        return None

    def is_full(self):
        return np.all(self.board != 0)

    def get_valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

# Neural Network
def create_network():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(3, 3, 1)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='tanh')
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

# Monte Carlo Tree Search
class MCTS:
    def __init__(self, model, num_simulations=100):
        self.model = model
        self.num_simulations = num_simulations

    def search(self, state):
        for _ in range(self.num_simulations):
            self.simulate(state)
        
        moves, _ = zip(*state.get_valid_moves())
        move_probs = np.zeros(9)
        for move in moves:
            move_probs[move] = self.get_visit_count(state, move)
        move_probs /= np.sum(move_probs)
        return move_probs

    def simulate(self, state):
        # Simplified MCTS simulation (random playouts)
        temp_state = TicTacToe()
        temp_state.board = state.board.copy()
        temp_state.current_player = state.current_player
        
        while True:
            winner = temp_state.check_winner()
            if winner is not None or temp_state.is_full():
                return -winner if winner else 0
            
            valid_moves = temp_state.get_valid_moves()
            action = random.choice(valid_moves)
            temp_state.step(action[0] * 3 + action[1])

    def get_visit_count(self, state, action):
        # Simplified: return a random value (in a full implementation, this would return actual visit counts)
        return random.random()

# Training loop
def train(num_episodes=1000):
    env = TicTacToe()
    model = create_network()
    mcts = MCTS(model)
    
    win_rates = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action_probs = mcts.search(env)
            action = np.argmax(action_probs)
            next_state, winner, done = env.step(action)
            
            if winner is not None:
                win_rates.append(1 if winner == 1 else 0)
            elif done:
                win_rates.append(0.5)  # Draw
            
            state = next_state
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Win rate: {np.mean(win_rates[-100:]):.2f}")
    
    return win_rates

# Run training
win_rates = train(num_episodes=1000)

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(range(len(win_rates)), win_rates)
plt.title('AlphaZero-inspired Agent Learning Progress')
plt.xlabel('Episode')
plt.ylabel('Win Rate (Moving Average)')
plt.ylim(0, 1)
plt.show()