import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(n_steps, dt):
    """
    Simulate n_steps steps of Brownian motion.
    
    Parameters:
    n_steps (int): Number of time steps to simulate.
    dt (float): Time step size.
    
    Returns:
    x (numpy array): The simulated Brownian motion path.
    """
    # Initialize the random walk with a single point at the origin
    x = np.zeros(n_steps + 1)
    
    # For each step, generate a new random value and add it to the position
    for i in range(1, n_steps + 1):
        x[i] = x[i - 1] + (np.random.uniform(-1, 1) * np.sqrt(dt))
    
    return x

# Define simulation parameters
n_steps = int(1e5)
dt = 0.001
t_max = n_steps * dt

# Simulate Brownian motion and plot the result
x = brownian_motion(n_steps, dt)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(t_max) * dt, x)
plt.title('Brownian Motion')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.show()