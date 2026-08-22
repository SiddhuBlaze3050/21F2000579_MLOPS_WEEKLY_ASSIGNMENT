import pandas as pd
import json
import os

def generate_v1_raw_jsonl(input_csv, output_jsonl):
    """
    Converts tabular IRIS data into Vertex AI JSONL format (Raw Features).
    """
    print(f"Generating V1 (Raw Features) from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    records = []
    for _, row in df.iterrows():
        input_text = (f"sepal_length: {row['sepal_length']}, "
                      f"sepal_width: {row['sepal_width']}, "
                      f"petal_length: {row['petal_length']}, "
                      f"petal_width: {row['petal_width']}")
        output_text = str(row['species'])
        
        records.append({"input_text": input_text, "output_text": output_text})
        
    with open(output_jsonl, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
            
    print(f"✅ V1 Created: {output_jsonl} ({len(records)} records)")

def generate_v2_desc_jsonl(input_csv, output_jsonl):
    """
    Converts tabular IRIS data into Vertex AI JSONL format (Natural Language).
    """
    print(f"Generating V2 (Natural Language Description) from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    records = []
    for _, row in df.iterrows():
        # 1. Format the input string as a natural, conversational sentence
        input_text = (f"A flower specimen has a sepal length of {row['sepal_length']} cm, "
                      f"sepal width of {row['sepal_width']} cm, "
                      f"petal length of {row['petal_length']} cm, "
                      f"and petal width of {row['petal_width']} cm. "
                      f"Identify the iris species.")
        
        # 2. Format the output string as a complete sentence
        output_text = f"This is Iris {row['species']}."
        
        # 3. Create the Vertex AI required dictionary
        records.append({"input_text": input_text, "output_text": output_text})
        
    # 4. Write to JSONL file
    with open(output_jsonl, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
            
    print(f"✅ V2 Created: {output_jsonl} ({len(records)} records)")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    input_file = "data/iris.csv"
    v1_output_file = "data/v1_raw.jsonl"
    v2_output_file = "data/v2_desc.jsonl"
    
    # Run both generators
    generate_v1_raw_jsonl(input_file, v1_output_file)
    generate_v2_desc_jsonl(input_file, v2_output_file)