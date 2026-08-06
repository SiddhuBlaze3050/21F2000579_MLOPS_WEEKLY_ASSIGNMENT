import pandas as pd
import numpy as np
import os

# Set a random seed for reproducible poisoning across runs
np.random.seed(42)

def generate_poisoned_data(input_csv, output_csv, corruption_ratio):
    # Load the clean baseline dataset
    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv}. Check your data directory.")
        return

    df = pd.read_csv(input_csv)
    
    # Calculate the exact number of rows to poison
    n_samples = len(df)
    n_poisoned = int(n_samples * corruption_ratio)
    
    # Randomly select rows to corrupt (without replacement)
    poison_indices = np.random.choice(df.index, size=n_poisoned, replace=False)
    
    # Identify feature columns and the target column 
    # (Assuming the target/label is the last column in the CSV)
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    
    # Extract unique classes to assign random labels
    unique_classes = df[target_col].unique()
    
    for idx in poison_indices:
        # 1. Corrupt Features: Inject random noise bounded by the dataset's min/max
        for col in feature_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            df.at[idx, col] = np.random.uniform(min_val, max_val)
            
        # 2. Corrupt Label: Assign a random class
        df.at[idx, target_col] = np.random.choice(unique_classes)
        
    # Save the poisoned variant to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Created {output_csv} | Poisoned {n_poisoned} out of {n_samples} samples ({int(corruption_ratio*100)}%).")

if __name__ == "__main__":
    # Define the path to your clean data
    input_file = "data/iris.csv"  
    
    # Generate the three poisoned variants
    generate_poisoned_data(input_file, "data/iris_poison_5.csv", 0.05)
    generate_poisoned_data(input_file, "data/iris_poison_10.csv", 0.10)
    generate_poisoned_data(input_file, "data/iris_poison_50.csv", 0.50)