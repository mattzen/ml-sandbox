import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate synthetic binary classification data
np.random.seed(0)
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Generate new da`ta points for predictions
X_new = np.linspace(-3, 3, 300).reshape(-1, 1)
y_prob = model.predict_proba(X_new)[:, 1]  # Get the probability for class 1

# Plot the data points and the logistic regression curve
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_new, y_prob, color='red', linewidth=2, label='Logistic Regression Curve')
plt.xlabel('X')
plt.ylabel('Probability of class 1')
plt.title('Logistic Regression Example')
plt.legend()
plt.show()