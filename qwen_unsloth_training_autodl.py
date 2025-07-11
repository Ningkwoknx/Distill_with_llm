import torch
import pandas as pd
import numpy as np
import re
import argparse
from tqdm import tqdm

# Transformers and Unsloth imports
from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling, logging as hf_logging
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from modelscope import AutoModelForCausalLM, AutoTokenizer

# Metrics imports
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/.cache/huggingface"

# +-------------------------------------------------+
# |              1. HELPER FUNCTIONS                |
# +-------------------------------------------------+

def extract_floats_with_4_decimal_places(text):
    """Extracts the first float with 4 decimal places from a string."""
    pattern = r'[-+]?\d*\.\d{4}'
    matches = re.findall(pattern, text)
    if not matches:
        # Fallback for cases where the model might not produce exactly 4 decimal places
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

# +-------------------------------------------------+
# |              2. MAIN SCRIPT LOGIC               |
# +-------------------------------------------------+

def main(args):
    # --- A. Setup and Model Loading ---
    print(" Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    print(" Applying PEFT (LoRA) configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # --- B. Data Preparation ---
    # Define prompt structure (now it's part of the script)
    SYSTEM_MESSAGE = """You are a multilingual translation evaluation expert...""" # (Keep your full prompt here)
    def create_user_message(source_text, mt_text):
        return f"""### Source sentence:\n{source_text}\n\n### MT sentence:\n{mt_text}"""
    
    # Define the tokenization function using the prompts
    def tokenize(example):
        prompt_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": create_user_message(example['src'], example['mt'])},
        ]
        completion_messages = [{"role": "assistant", "content": f" {example['zmean']:.4f}"}]
        
        prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True, add_special_tokens=False)
        completion_ids = tokenizer.apply_chat_template(completion_messages, tokenize=True, add_special_tokens=False)
        
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        attention_mask = [1] * len(input_ids)
        
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(attention_mask)}

    print(f" Loading data from paths: \n\tTrain: {args.train_file}\n\tDev: {args.dev_file}\n\tTest: {args.test_file}")
    traindf = pd.read_csv(args.train_file, sep="\t")
    devdf = pd.read_csv(args.dev_file, sep="\t")
    testdf = pd.read_csv(args.test_file, sep="\t")

    train_dataset_hf = Dataset.from_pandas(traindf)
    tokenized_train = train_dataset_hf.map(
        tokenize,
        remove_columns=train_dataset_hf.column_names
    )

    dev_dataset_hf = Dataset.from_pandas(devdf)
    tokenized_dev = dev_dataset_hf.map(
        tokenize,
        remove_columns=dev_dataset_hf.column_names
    )
    
    #The following two lines work in google colab but not in autodl with python scripts
    # train_dataset = Dataset.from_pandas(traindf).map(tokenize, remove_columns=traindf.column_names)
    # dev_dataset = Dataset.from_pandas(devdf).map(tokenize, remove_columns=devdf.column_names)

    # --- C. Training ---
    print(" Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            bf16=True,
            optim="paged_adamw_32bit",
            num_train_epochs=3,
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            logging_steps=50,
            warmup_steps=100,
            learning_rate=2e-5,
            completion_only_loss=True,
            group_by_length=True,
            report_to="none",
        )
    )
    
    print(" Starting training...")
    trainer.train()
    print(" Training finished!")

    # --- D. Evaluation ---
    print("\n Starting evaluation on the test set...")
    hf_logging.set_verbosity_error() # Make the progress bar clean
    
    predicted_scores = []
    ground_truth_scores = []

    for _, row in tqdm(testdf.iterrows(), total=len(testdf), desc="Evaluating"):
        prompt_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": create_user_message(row['src'], row['mt'])},
        ]
        
        formatted_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(formatted_text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
        
        decoded_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted_score = extract_floats_with_4_decimal_places(decoded_text)
        
        predicted_scores.append(predicted_score)
        ground_truth_scores.append(row['zmean'])

    # --- E. Results and Saving ---
    valid_indices = [i for i, score in enumerate(predicted_scores) if score is not None]
    cleaned_predictions = [predicted_scores[i] for i in valid_indices]
    cleaned_ground_truth = [ground_truth_scores[i] for i in valid_indices]
    
    print(f"\nSuccessfully predicted scores for {len(cleaned_predictions)}/{len(predicted_scores)} samples.")

    if len(cleaned_predictions) > 1:
        calculate_and_print_metrics(cleaned_ground_truth, cleaned_predictions)
        
        output_file_path = f"{args.output_dir}/test_predictions.tsv"
        save_predictions_to_tsv(testdf, predicted_scores, output_file_path)
    else:
        print("\nNot enough valid predictions to calculate metrics or save results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a Qwen model for translation quality estimation.")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit", help="Name of the model to fine-tune.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training TSV file.")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to the development/validation TSV file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test TSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final predictions.")
    
    args = parser.parse_args()
    main(args)