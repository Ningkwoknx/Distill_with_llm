#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qwen Generative Fine-tuning Script
This script fine-tunes a Qwen model for a generative regression task.
The model learns to generate a score as text.
"""

# ==================== 1. Imports ====================
import os
os.environ["USE_MPI"] = "0"
import torch
torch.cuda.empty_cache()
import pandas as pd
import numpy as np
import pickle
import traceback
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sklearn.preprocessing import StandardScaler

import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

print(f"Current total memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
print(f"Current available memory: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
torch.cuda.empty_cache()

# ==================== 2. Global Configurations ====================
MODEL_PATH = "Qwen/Qwen3-0.6B"
TRAIN_FILE = "../data/toy_train.tsv"
DEV_FILE = "../data/toy_dev.tsv"
OUTPUT_DIR = "../models/outdir/qwen_generative_baseline_toy" # Use a new directory
MAX_LENGTH = 512

# ==================== 3. Helper Functions ====================
def create_deepspeed_config():
    """Configurations for training with deepspeed"""
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "steps_per_print": 200,
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 5e8
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "fp16": {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 2,
            "synchronize_checkpoint_boundary": True,
            "profile": False
        },
        "gradient_clipping": 1.0,
        "comms_logger": {
            "enabled": False
        },
        "memory_breakdown": {
            "enabled": False
        }
    }
    return config

def save_deepspeed_config(config, output_dir):
    """Save deepspeed configurations to files"""
    config_path = os.path.join(output_dir, "ds_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path

def load_tsv_data(file_path, label_column='zmean'):
    # (This is a simplified version for clarity)
    print(f"\nLoading data from: {file_path}")
    encoding = 'latin-1' if 'train' in file_path else 'utf-8'
    df = pd.read_csv(file_path, sep='\t', encoding=encoding)
    src_texts = df['src'].astype(str).tolist()
    mt_texts = df['mt'].astype(str).tolist()
    targets = df[label_column].astype(float).tolist()
    print(f"Loaded {len(df)} samples.")
    return src_texts, mt_texts, targets

def create_prompt_for_generation(src_text, mt_text):
    """Creates the input prompt for the generative model."""
    return f"""You are a multilingual translation evaluation expert. Your task is to predict the quality score for translation pairs.

Scoring Criteria:
-2.0 to -1.0: Poor or very poor translation with meaning deviation or severe errors.
-1.0 to 0.0: Flawed translation but understandable.
0.0 to 1.0: Good translation in general.
1.0 to 2.0: Excellent or flawless translation.

Source text: "{src_text}"
Translation to evaluate: "{mt_text}"
Output: ONLY the numerical score (e.g., -1.2345).
Score:"""

# ==================== 4. Dataset Class Definition ====================
class GenerativeRegressionDataset(Dataset):
    """Prepares data for the generative regression task."""
    def __init__(self, src_texts, mt_texts, final_targets, tokenizer):
        self.src_texts = src_texts
        self.mt_texts = mt_texts
        self.final_targets = final_targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        prompt = create_prompt_for_generation(self.src_texts[idx], self.mt_texts[idx])
        label_text = f"{self.final_targets[idx].item():.4f}"
        full_text = prompt + label_text + self.tokenizer.eos_token
        
        model_inputs = self.tokenizer(full_text, max_length=MAX_LENGTH, truncation=True, padding=False)
        model_inputs["labels"] = model_inputs["input_ids"][:]
        return model_inputs

# ==================== 5. The main() Function ====================
def main():
    """The complete and correct training workflow."""
    
    # --- Module 1: Env & Tokenizer Setup ---
    print("--- 1. Environment and Tokenizer Setup ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--- 1.1. Creating DeepSpeed Configuration ---")
    ds_config = create_deepspeed_config()
    ds_config_path = save_deepspeed_config(ds_config, OUTPUT_DIR)
    print(f"DeepSpeed config saved to: {ds_config_path}")


    # --- Module 2: Data Loading & Normalization (StandardScaler + Clip) ---
    print("\n--- 2. Loading and Processing Data ---")
    train_src, train_mt, train_targets_raw = load_tsv_data(TRAIN_FILE)
    dev_src, dev_mt, dev_targets_raw = load_tsv_data(DEV_FILE)
    
    scaler = StandardScaler()
    scaler.fit(np.array(train_targets_raw).reshape(-1, 1))
    
    train_targets_std = scaler.transform(np.array(train_targets_raw).reshape(-1, 1))
    dev_targets_std = scaler.transform(np.array(dev_targets_raw).reshape(-1, 1))
    
    train_targets_final = np.clip(train_targets_std, -2, 2)
    dev_targets_final = np.clip(dev_targets_std, -2, 2)
    
    scaler_path = os.path.join(OUTPUT_DIR, "standard_scaler_for_clipping.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # --- Module 3: Dataset Creation ---
    print("\n--- 3. Creating Datasets ---")
    train_dataset = GenerativeRegressionDataset(train_src, train_mt, train_targets_final, tokenizer)
    dev_dataset = GenerativeRegressionDataset(dev_src, dev_mt, dev_targets_final, tokenizer)

    # --- Module 4: Model Loading ---
    print("\n--- 4. Loading Model ---")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # --- Module 5: Trainer Configuration ---
    print("\n--- 5. Configuring Trainer ---")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
	per_device_train_batch_size = 4,
	gradient_accumulation_steps = 4,
	fp16 = True,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=2000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
	logging_steps=10,
        report_to="none",
	logging_first_step=True,
	disable_tqdm=False,
        deepspeed=ds_config_path,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        label_pad_token_id=-100, # Important for correct loss calculation
        pad_to_multiple_of=8     # Optional: for performance on modern GPUs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
    )

    # --- Module 6: Execute Training ---
    print("\n--- 6. Starting Training ---")
    trainer.train()

    # --- Module 7: Final Saving ---
    print("\n--- 7. Saving Final Model ---")
    final_model_path = os.path.join(OUTPUT_DIR, "final_best_model")
    os.makedirs(final_model_path, exist_ok=True)  
    model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    model_to_save.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final best model and tokenizer saved to: {final_model_path}")
    print("\n--- Workflow Finished ---")

# ==================== 6. The Script Entry Point ====================
if __name__ == "__main__":
    main()

