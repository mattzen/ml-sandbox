import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (assuming we have a CSV file with house data)
data = pd.read_csv('house_data.csv')

# Select features and target variable
X = data[['sqft_living', 'bedrooms', 'bathrooms', 'floors']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Example of predicting a house price
new_house = np.array([[2000, 3, 2, 1]])  # 2000 sqft, 3 bedrooms, 2 bathrooms, 1 floor
predicted_price = model.predict(new_house)
print(f"Predicted price for the new house: ${predicted_price[0]:,.2f}")

# Print the coefficients to understand feature importance
for feature, coef in zip(X.columns, model.coef_):
    print(f"Coefficient for {feature}: {coef}")