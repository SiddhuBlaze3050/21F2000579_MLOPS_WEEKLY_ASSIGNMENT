import pandas as pd
import json
import os

def generate_v1_raw_jsonl(input_csv, output_jsonl):
    """
    Converts tabular IRIS data into Vertex AI JSONL format (Raw Features).
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    records = []
    for _, row in df.iterrows():
        # 1. Format the input string exactly as requested
        input_text = (f"sepal_length: {row['sepal_length']}, "
                      f"sepal_width: {row['sepal_width']}, "
                      f"petal_length: {row['petal_length']}, "
                      f"petal_width: {row['petal_width']}")
        
        # 2. Format the output string
        output_text = str(row['species'])
        
        # 3. Create the Vertex AI required dictionary
        records.append({
            "input_text": input_text,
            "output_text": output_text
        })
        
    # 4. Write to JSONL file
    with open(output_jsonl, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
            
    print(f"✅ Successfully created {output_jsonl} with {len(records)} records.")

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Define paths
    input_file = "data/iris.csv"
    v1_output_file = "data/v1_raw.jsonl"
    
    # Run the V1 generator
    generate_v1_raw_jsonl(input_file, v1_output_file)