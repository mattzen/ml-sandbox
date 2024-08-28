import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Number of houses
n_houses = 1000

# Generate synthetic data
sqft_living = np.random.randint(1000, 5000, n_houses)
bedrooms = np.random.randint(1, 6, n_houses)
bathrooms = np.random.randint(1, 5, n_houses)
floors = np.random.choice([1, 2, 3], n_houses, p=[0.5, 0.4, 0.1])

# Create a base price
base_price = 100000

# Generate prices with some randomness
prices = (
    base_price +
    sqft_living * 100 +
    bedrooms * 15000 +
    bathrooms * 10000 +
    floors * 25000 +
    np.random.normal(0, 50000, n_houses)  # Add some noise
)

# Ensure all prices are positive
prices = np.maximum(prices, 0)

# Create a DataFrame
df = pd.DataFrame({
    'sqft_living': sqft_living,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'floors': floors,
    'price': prices.astype(int)
})

# Save to CSV
df.to_csv('house_data.csv', index=False)

print("Sample of the generated data:")
print(df.head())

print("\nDataset summary:")
print(df.describe())

print("\nData saved to 'house_data.csv'")