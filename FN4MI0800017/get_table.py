import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('training_log.csv')

# Filter the rows where the iteration column is divisible by 50,000
filtered_rows = df[df['iteration'] % 50000 == 0]

# Print the filtered rows
print(filtered_rows.to_string(index=False))
