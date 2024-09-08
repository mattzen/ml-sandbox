import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('house_data.csv')

# Select features and target variable
X = data[['sqft_living', 'bedrooms', 'bathrooms', 'floors']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
fig.suptitle('Linear Regression: House Price Predictions', fontsize=16)

features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors']
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    
    # Scatter plot of actual data
    axs[row, col].scatter(X_test[feature], y_test, color='blue', alpha=0.5, label='Actual')
    
    # Sort the data for a smooth line plot
    sort_idx = X_test[feature].argsort()
    axs[row, col].plot(X_test[feature].iloc[sort_idx], y_pred[sort_idx], color='red', label='Predicted')
    
    axs[row, col].set_xlabel(feature)
    axs[row, col].set_ylabel('Price')
    axs[row, col].legend()
    axs[row, col].set_title(f'Price vs {feature}')

# Add text with model performance
plt.figtext(0.5, 0.01, f'Mean Squared Error: {mse:.2f}\nR-squared Score: {r2:.2f}', 
            ha='center', fontsize=12)

# Adjust layout and save
plt.tight_layout()
plt.savefig('house_price_regression.png')
print("Visualization saved as 'house_price_regression.png'")

# Display feature importances
importances = pd.DataFrame({'feature': features, 'importance': model.coef_})
importances = importances.sort_values('importance', ascending=False)
print("\nFeature Importances:")
print(importances)


