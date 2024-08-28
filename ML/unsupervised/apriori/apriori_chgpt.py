# Install necessary packages if you haven't already
# !pip install mlxtend pandas matplotlib

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Sample transaction dataset
data = {'Bread': [1, 0, 1, 1, 0],
        'Milk': [1, 1, 1, 0, 1],
        'Beer': [0, 1, 0, 1, 1],
        'Diapers': [1, 1, 1, 1, 0],
        'Eggs': [0, 1, 0, 1, 1]}

df = pd.DataFrame(data)

# Generate frequent itemsets with a minimum support threshold of 0.6
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Generate association rules with a minimum confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print(frequent_itemsets)
print(rules)

# Visualization of the support of frequent itemsets
plt.figure(figsize=(8,6))
plt.bar(x = list(range(len(frequent_itemsets))),
        height = frequent_itemsets['support'], color='skyblue')
plt.xticks(list(range(len(frequent_itemsets))),
           ['+'.join(item) for item in frequent_itemsets['itemsets']],
           rotation=90)
plt.ylabel('Support')
plt.xlabel('Itemsets')
plt.title('Frequent Itemsets')
plt.show()

# Visualization of confidence vs lift for the association rules
plt.figure(figsize=(8,6))
plt.scatter(rules['confidence'], rules['lift'], alpha=0.7, marker="o")
plt.title('Association Rules - Confidence vs Lift')
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.grid(True)
plt.show()
