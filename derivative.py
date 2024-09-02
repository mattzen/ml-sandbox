import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt

def calculate_derivative(func, x0, dx=1e-7):
    return derivative(func, x0, dx)

# Define the function for which we want to calculate the derivative
def func(x):
    return x**2 + 3*x - 4

x = np.linspace(-10, 10, 100)  # Generate an array of x values between -10 and 10
y = func(x)  # Calculate the corresponding y values

# Calculate the derivative at various points
derivative_values = []
for i in range(len(x)):
    d_value = calculate_derivative(func, x[i])
    derivative_values.append(d_value)

# Plot the functionclear
plt.figure(figsize=(8,6))
plt.plot(x, y, label='f(x) = x^2 + 3x - 4')
plt.title('Function and its Derivative')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

# Plot the derivative
plt.plot(x, derivative_values, label='Derivative of f(x)', linestyle='--', marker='o')

# Set labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and its Derivative')

# Show the plot
print("Plotting function and derivative...")
plt.show()