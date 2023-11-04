import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import subprocess


# Import data
df = pd.read_csv('non-encoded.csv')

categorical_column = 'Education'
numerical_column = 'total_num_of_purchases'
# Create a box plot to visualize the relationship
plt.figure(figsize=(10, 6))
sns.boxplot(x=categorical_column, y=numerical_column, data=df, palette='Set2')
plt.title(f'Box Plot: {categorical_column} vs. {numerical_column}')
plt.xlabel(categorical_column)
plt.ylabel(numerical_column)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)

# Save the figure
plt.savefig('vis.png', bbox_inches='tight')