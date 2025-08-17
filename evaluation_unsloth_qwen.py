# evaluate_model.py
import torch
import pandas as pd
import numpy as np
import re
import argparse
from tqdm import tqdm

# Transformers and Unsloth imports
from unsloth import FastLanguageModel
from transformers import logging as hf_logging

# Metrics imports
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import os

# Set Hugging Face environment variables for mirror and cache
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/.cache/huggingface"

# +-------------------------------------------------+
# |              1. HELPER FUNCTIONS                |
# |      (Copied from your training script)         |
# +-------------------------------------------------+

def extract_floats_with_4_decimal_places(text):
    """Extracts the first float with 4 decimal places from a string."""
    pattern = r'[-+]?\d*\.\d{4}'
    matches = re.findall(pattern, text)
    if not matches:
        matches = re.findall(r'[-+]?\d+\.\d+', text)
    return float(matches[0]) if matches else None

def calculate_and_print_metrics(ground_truth, predictions):
    """Calculates and prints a series of evaluation metrics."""
    print("\n--- Comprehensive Performance Metrics ---")
    pearson_corr, p_value = pearsonr(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    mse = mean_squared_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Pearson r         : {pearson_corr:.4f}")
    print(f"P-value           : {p_value:.4e}")
    print(f"MAE (Abs. Error)  : {mae:.4f}")
    print(f"MSE (Squared Err) : {mse:.4f}")
    print(f"RMSE (Root Sq. Err): {rmse:.4f}")
    print("---------------------------------------")

def save_predictions_to_tsv(original_dataframe, predictions, output_path):
    """Saves the original data with prediction scores to a TSV file."""
    df_with_predictions = original_dataframe.copy()
    df_with_predictions['predicted_zmean'] = predictions
    df_with_predictions.to_csv(output_path, sep='\t', index=False)
    print(f"\nPredictions successfully saved to: {output_path}")

# --- Prompt formatting functions (also needed for evaluation) ---
SYSTEM_MESSAGE = """You are a multilingual translation evaluation expert. Your task is to predict the quality score for translation pairs.
The quality score is a number between -2.0 and 2.0 and could fall into the following categories:
Scoring Criteria:
-2.0 to -1.0: Poor or very poor translation with meaning deviation or severe errors.
-1.0 to 0.0: Flawed translation but understandable.
0.0 to 1.0: Good translation in general.
1.0 to 2.0: Excellent or flawless translation.
The relative position of the scores in this range indicates subtle differences in translation quality. Based on the provided source, translation, and reasoning, output only the numerical score.""" 

def create_user_message(source_text, mt_text, gpt_explanation):
    return f"""### Source sentence:\n{source_text}\n\n### MT sentence:\n{mt_text}\n\n### GPT reasoning:\n
{gpt_explanation}"""


# +-------------------------------------------------+
# |           2. MAIN EVALUATION LOGIC              |
# +-------------------------------------------------+

def main(args):
    # --- A. Load the FINE-TUNED Model ---
    # This is the key change: we load from a local path, not the Hugging Face Hub.
    print(f"Loading fine-tuned model from: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path, # Load from the local directory
        dtype=None,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    # Note: No PEFT configuration needed here, Unsloth handles it automatically when loading.

    # --- B. Load Test Data ---
    print(f"Loading test data from: {args.test_file}")
    testdf = pd.read_csv(args.test_file, sep="\t")

    # --- C. Run Inference Loop ---
    print(f"\nStarting evaluation on {len(testdf)} test samples...")
    hf_logging.set_verbosity_error() # Make the progress bar clean
    
    predicted_scores = []
    ground_truth_scores = []

    for _, row in tqdm(testdf.iterrows(), total=len(testdf), desc="Evaluating"):
        prompt_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": create_user_message(row['src'], row['mt'], row['gpt_explanation'])},
        ]
        
        # Use the tokenizer to create the formatted prompt string
        formatted_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_text, return_tensors="pt").to(model.device)
        
        # Generate the output from the model
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=10, # Needs to be enough for a score like "-1.2345"
            do_sample=False,   # For evaluation, greedy search is often better
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode and extract the numerical score
        decoded_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted_score = extract_floats_with_4_decimal_places(decoded_text)
        
        predicted_scores.append(predicted_score)
        ground_truth_scores.append(row['zmean'])

    # --- D. Calculate Metrics and Save Results ---
    valid_indices = [i for i, score in enumerate(predicted_scores) if score is not None]
    cleaned_predictions = [predicted_scores[i] for i in valid_indices]
    cleaned_ground_truth = [ground_truth_scores[i] for i in valid_indices]
    
    print(f"\nSuccessfully predicted scores for {len(cleaned_predictions)}/{len(predicted_scores)} samples.")

    if len(cleaned_predictions) > 1:
        calculate_and_print_metrics(cleaned_ground_truth, cleaned_predictions)
        save_predictions_to_tsv(testdf, predicted_scores, args.output_file)
    else:
        print("\nNot enough valid predictions to calculate metrics or save results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Qwen model.")
    # --- New, simplified arguments for evaluation ---
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the fine-tuned model checkpoint.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test TSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final predictions TSV file.")
    
    args = parser.parse_args()
    main(args)