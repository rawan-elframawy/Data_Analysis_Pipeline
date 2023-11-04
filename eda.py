import pandas as pd
import sys
import subprocess


def get_dominant_dtype():
    # Determine the data types of each column
    data_types = df.dtypes

    # Count the frequency of each data type
    type_counts = data_types.value_counts()

    # Find the most common data type
    most_common_type = type_counts.idxmax()

    # Save insight
    with open('eda-in-1.txt', 'w') as f:
        f.write(f'The dominant dtype is {most_common_type}')


def get_stats():
    summary_stats = df.describe()
    with open('eda-in-2.txt', 'w') as f:
        f.write(summary_stats.to_string())


def marital_stats():
    with open('eda-in-3.txt', 'w') as f:
        f.write(df['Marital_Status'].value_counts().to_string())
        

# Import data
df = pd.read_csv(sys.argv[1], sep='\t')

# 3 simple insights
get_dominant_dtype()
get_stats()
marital_stats()

# Invoke next file
subprocess.run(['python', 'vis.py', sys.argv[1]])