import os
import gc
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model_path, version="V1"):
    print(f"\n==========================================")
    print(f" Evaluating {version} Model from: {model_path}")
    print(f"==========================================")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32
    )
    model.eval()

    X_test = pd.read_csv("data/X_test.csv")
    y_test_df = pd.read_csv("data/y_test.csv")
    y_true = y_test_df["species"].str.strip().str.lower().tolist()

    valid_classes = {"setosa", "versicolor", "virginica"}
    predictions = []
    compliant_count = 0

    concept_correct = 0 # Track if it got the species right at all
    
    for idx, row in X_test.iterrows():
        expected_species = y_true[idx]
        
        if version == "V1":
            user_text = (
                f"sepal_length: {row['sepal_length']}, "
                f"sepal_width: {row['sepal_width']}, "
                f"petal_length: {row['petal_length']}, "
                f"petal_width: {row['petal_width']}"
            )
        else:
            user_text = (
                f"A flower specimen has a sepal length of {row['sepal_length']} cm, "
                f"sepal width of {row['sepal_width']} cm, "
                f"petal length of {row['petal_length']} cm, "
                f"and petal width of {row['petal_width']} cm. "
                f"Identify the iris species."
            )

        messages = [
            {"role": "system", "content": "Classify the flower based on its measurements into one of the following species: [Setosa, Versicolor, Virginica]"},
            {"role": "user", "content": user_text}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        raw_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        clean_output = raw_output.lower().replace(".", "").strip()
        
        # --- NEW: Concept Accuracy Check ---
        if expected_species in clean_output:
            concept_correct += 1

        # Format Compliance Check
        if version == "V2" and clean_output.startswith("this is iris "):
            clean_output = clean_output.replace("this is iris ", "").strip()

        if clean_output in valid_classes:
            compliant_count += 1
            predictions.append(clean_output)
        else:
            predictions.append("format_failure")

    total_samples = len(y_true)
    compliance_rate = (compliant_count / total_samples) * 100
    strict_accuracy = accuracy_score(y_true, predictions)
    concept_accuracy = (concept_correct / total_samples) * 100

    print(f"\nResults for {version}:")
    print(f"• Total Samples: {total_samples}")
    print(f"• Format Compliance Rate: {compliance_rate:.2f}% (Strict formatting)")
    print(f"• Strict Accuracy: {strict_accuracy * 100:.2f}%")
    print(f"• Concept Accuracy: {concept_accuracy:.2f}% (Did it guess the right species in the text?)")
    print("\nClassification Report:")
    print(classification_report(y_true, predictions, zero_division=0))
    
    print("🧹 Clearing model from RAM...")
    del model
    del tokenizer
    gc.collect()

if __name__ == "__main__":
    # v1_dir = "models/v1_raw"
    v2_dir = "models/v2_desc"
    
    # if os.path.exists(v1_dir):
    #     evaluate_model(v1_dir, version="V1")
    if os.path.exists(v2_dir):
        evaluate_model(v2_dir, version="V2")