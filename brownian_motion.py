import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 100)
ax.set_ylim(-10, 10)
ax.set_xlabel('Time')
ax.set_ylabel('Position')
ax.set_title('Simple Brownian Motion')

# Initialize particles
num_particles = 5
lines = [ax.plot([], [], label=f'Particle {i+1}')[0] for i in range(num_particles)]
ax.legend()

# Initialize data
t = np.linspace(0, 100, 100)
positions = np.zeros((num_particles, 100))

# Animation function
def animate(frame):
    for i in range(num_particles):
        if frame > 0:
            positions[i, frame] = positions[i, frame-1] + np.random.normal(0, 0.5)
        lines[i].set_data(t[:frame+1], positions[i, :frame+1])
    return lines

# Create animation
anim = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)

plt.show()