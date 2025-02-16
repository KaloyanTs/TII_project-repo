import csv

# Define the path to your CSV file
csv_file_path = 'results/training_log.csv'

# Initialize a counter for rows
row_count = 0

# Open the CSV file
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
    print(f"Reading from file: {csv_file_path}")
    reader = csv.reader(file)
    
    # Read the header row
    headers = next(reader)
    print(f"Headers: {headers}")
    
    # Iterate over the rows in the CSV file
    for row in reader:
        row_count += 1
        
        # Print every 50,000th row
        if int(row[headers.index('iteration')]) % 5000 == 0:
            print(f"{row}")
    
    # Print the last row of the file
    print(f"Last row: {row}")
