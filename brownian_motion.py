import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# 
class BrownianMotion:
    def __init__(self, num_particles=100, dimensions=2, bounds=(-10, 10)):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
    
    def update(self):
        self.positions += np.random.normal(0, 0.1, self.positions.shape)
        self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])

def animate(frame):
    brownian.update()
    scatter.set_offsets(brownian.positions)
    return scatter,

# Set up the simulation
num_particles = 100
brownian = BrownianMotion(num_particles)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(brownian.bounds)
ax.set_ylim(brownian.bounds)
ax.set_title("Brownian Motion Simulation")
scatter = ax.scatter(brownian.positions[:, 0], brownian.positions[:, 1], s=10)

# Create the animation
anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)

plt.show()