import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
file_path = "trainmap.csv"
df = pd.read_csv(file_path)

# Extract unique diseases from the 'disease' column
unique_diseases = df['disease'].unique()

# Create dictionaries to store indices for train and test sets for each disease
train_indices_dict = {}
test_indices_dict = {}

# Loop through each unique disease
for disease in unique_diseases:
    # Filter rows for the current disease
    disease_rows = df[df['disease'] == disease]
    
    # Get the indices of the rows for the current disease
    indices = disease_rows.index.tolist()
    
    # Split indices into train and test sets
    train_indices, test_indices = train_test_split(indices, test_size=0.15, random_state=42)
    
    # Store the indices in dictionaries
    train_indices_dict[disease] = train_indices
    test_indices_dict[disease] = test_indices

# Print or use the dictionaries as needed
print("Train Indices:")
print(train_indices_dict)
print("\nTest Indices:")
print(test_indices_dict)
