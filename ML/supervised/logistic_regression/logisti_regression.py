import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate some sample data
np.random.seed(0)
num_samples = 100
exam1_scores = np.random.randn(num_samples) * 10 + 70
exam2_scores = np.random.randn(num_samples) * 10 + 75
X = np.column_stack((exam1_scores, exam2_scores))

# Generate labels (admitted or not)
y = (exam1_scores + exam2_scores > 150).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                       np.arange(x2_min, x2_max, 0.1))
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4, cmap='viridis')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Logistic Regression Decision Boundary')
plt.colorbar()
plt.show()

# Function to predict admission for new students
def predict_admission(exam1_score, exam2_score):
    prediction = model.predict([[exam1_score, exam2_score]])
    probability = model.predict_proba([[exam1_score, exam2_score]])
    return prediction[0], probability[0][1]

# Example usage
new_student_exam1 = 85
new_student_exam2 = 90
prediction, probability = predict_admission(new_student_exam1, new_student_exam2)
print(f"\nNew student with Exam 1 score {new_student_exam1} and Exam 2 score {new_student_exam2}:")
print(f"Prediction: {'Admitted' if prediction == 1 else 'Not Admitted'}")
print(f"Probability of admission: {probability:.2f}")