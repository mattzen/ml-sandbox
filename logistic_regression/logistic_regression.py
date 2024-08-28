import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate some sample data
np.random.seed(0)
num_samples = 1000
test_scores = np.random.normal(loc=70, scale=10, size=(num_samples, 2))
admitted = (test_scores[:, 0] + test_scores[:, 1] > 150).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(test_scores, admitted, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the decision boundary
x_min, x_max = test_scores[:, 0].min() - 1, test_scores[:, 0].max() + 1
y_min, y_max = test_scores[:, 1].min() - 1, test_scores[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(test_scores[:, 0], test_scores[:, 1], c=admitted, alpha=0.8)
plt.xlabel("Test Score 1")
plt.ylabel("Test Score 2")
plt.title("Logistic Regression Decision Boundary")
plt.show()