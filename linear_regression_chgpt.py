import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 random points for X
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + Gaussian noise




# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict values
X_new = np.array([[0], [2]])  # We predict for a range of X values
y_predict = model.predict(X_new)

# Plot the data and the regression line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_new, y_predict, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()