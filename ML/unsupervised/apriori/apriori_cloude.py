import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# I've updated the code to create more meaningful visualizations. Here's what each new chart represents:

# Top 10 Association Rules by Lift:
# This horizontal bar chart shows the top 10 rules with the highest lift values. Lift measures how much more likely the consequent is to occur when the antecedent occurs, compared to its baseline probability. Higher lift values indicate stronger associations.
# Distribution of Support for Frequent Itemsets:
# This histogram shows the distribution of support values for the frequent itemsets. It helps you understand how common different itemsets are in the dataset.
# Support vs Confidence for Association Rules:
# This scatter plot shows the relationship between support and confidence for all generated rules. Each point represents a rule, with its position indicating its support (x-axis) and confidence (y-axis).
# Top 10 Frequent Itemsets:
# This bar chart shows the top 10 most frequent itemsets and their support values. It gives you a quick view of the most common combinations of items in the transactions.

# These visualizations should provide more insightful information about the association rules and frequent itemsets discovered by the Apriori algorithm.
# To interpret these charts:

# In the Top 10 Association Rules chart, longer bars indicate stronger associations between items.
# The Support Distribution histogram shows how many itemsets have different levels of support. A right-skewed distribution is common, indicating many itemsets with low support and fewer with high support.
# In the Support vs Confidence scatter plot, points in the upper-right corner represent rules with both high support and high confidence, which are generally the most interesting rules.
# The Top 10 Frequent Itemsets chart shows which combinations of items occur most often in the transactions.

# Generate sample data
np.random.seed(42)
n_transactions = 1000
items = ['apple', 'banana', 'orange', 'milk', 'bread', 'cheese', 'yogurt', 'eggs']

transactions = []
for _ in range(n_transactions):
    n_items = np.random.randint(1, 6)
    transaction = np.random.choice(items, n_items, replace=False)
    transactions.append(list(transaction))

# Convert transactions to one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Visualize top 10 rules by lift
plt.figure(figsize=(12, 6))
top_rules = rules.sort_values('lift', ascending=False).head(10)
plt.barh(range(len(top_rules)), top_rules['lift'])
plt.yticks(range(len(top_rules)), [', '.join(rule) for rule in top_rules['antecedents']])
plt.xlabel('Lift')
plt.title('Top 10 Association Rules by Lift')
plt.tight_layout()
plt.show()

# Visualize support distribution of frequent itemsets
plt.figure(figsize=(10, 6))
plt.hist(frequent_itemsets['support'], bins=20)
plt.xlabel('Support')
plt.ylabel('Count')
plt.title('Distribution of Support for Frequent Itemsets')
plt.show()

# Visualize confidence vs support for all rules
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence for Association Rules')
plt.show()

# Print top 5 rules by lift
print(rules.sort_values('lift', ascending=False).head())

# Visualize frequent itemsets
plt.figure(figsize=(12, 6))
top_itemsets = frequent_itemsets.sort_values('support', ascending=False).head(10)
plt.bar(top_itemsets['itemsets'].apply(lambda x: ', '.join(list(x))), top_itemsets['support'])
plt.xticks(rotation=90)
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Top 10 Frequent Itemsets')
plt.tight_layout()
plt.show()