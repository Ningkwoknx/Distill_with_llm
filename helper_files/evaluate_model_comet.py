#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Evaluation Script

This script loads a pre-trained prompt-enhanced Qwen regression model
and evaluates its performance on a given validation or test dataset.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import traceback
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm # For a nice progress bar

# ==============================================================================
#  1. CONFIGURATION: 请在这里修改您的路径
# ==============================================================================

# 指向您保存的最佳模型的 checkpoint 文件夹
# 例如: "/home/ubuntu/distillation_project/output/checkpoint-1200"
BASE_MODEL_PATH = "../models/qwen"

CHECKPOINT_PATH = "../models/outdir/qwen_on_comet/checkpoint-20800" # <--- 修改这里

# 指向您要用于验证的数据文件（可以是dev set或test set）
EVAL_FILE_PATH = "../data/zmean_test_10_with_teacher_scores.tsv" # <--- 修改这里

# 验证需要知道原始zmean的范围，这个范围必须从【训练集】中学习，以保证一致性
# 所以请提供原始训练文件的路径
TRAIN_FILE_PATH = "zmean_train_80_with_teacher_scores.tsv" # <--- 修改这里

# 其他配置
MAX_LENGTH = 512
EVAL_BATCH_SIZE = 8 # 可以根据您的显存大小调整

# ==============================================================================
#  2. NECESSARY CLASS AND FUNCTION DEFINITIONS
#  (从训练脚本中复制过来，确保模型可以被正确加载和使用)
# ==============================================================================

class PromptEnhancedQwenRegression(nn.Module):
    """通过prompt增强的Qwen回归模型 - 用回归头输出分数"""
    def __init__(self, model_path, freeze_layers=True, use_fp16=False):
        super().__init__()
        model_dtype = torch.float16 if use_fp16 else torch.float32
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            local_files_only=True
        )
        if hasattr(self.qwen.config, 'hidden_size'):
            hidden_size = self.qwen.config.hidden_size
        else:
            hidden_size = 1024
        self.evaluation_head = nn.Sequential(
            nn.Linear(hidden_size, 1024, dtype=model_dtype),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512, dtype=model_dtype),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256, dtype=model_dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1, dtype=model_dtype)
        )
    
    @staticmethod
    def create_evaluation_prompt(src_text, mt_text, true_score=None, for_training=True):
        if for_training and true_score is not None:
            prompt = f"""You are a multilingual translation evaluation expert. Your task is to predict the quality score for translation pairs.\n\nScoring Criteria:\n1.4 to 1.7: Excellent translation\n0.5 to 1.3: Good translation\n-0.5 to 0.4: Average translation\n-1.3 to -0.6: Poor translation\n-1.7 to -1.4: Very poor translation\n\nSource text: "{src_text}"\nTranslation to evaluate: "{mt_text}"\nGround truth score: {true_score:.4f}\nOutput: ONLY the numerical score (e.g., -1.2345).\nScore:"""
        else:
            prompt = f"""You are a multilingual translation evaluation expert. Your task is to predict the quality score for translation pairs.\n\nScoring Criteria:\n1.4 to 1.7: Excellent translation\n0.5 to 1.3: Good translation\n-0.5 to 0.4: Average translation\n-1.3 to -0.6: Poor translation\n-1.7 to -1.4: Very poor translation\n\nSource text: "{src_text}"\nTranslation to evaluate: "{mt_text}"\nOutput: ONLY the numerical score (e.g., -1.2345).\nScore:"""
        return prompt
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state.mean(dim=1)
        evaluation_score = self.evaluation_head(pooled_output.float())
        loss = None
        if labels is not None:
            labels = labels.float()
            loss_fn = nn.MSELoss()
            loss = loss_fn(evaluation_score.squeeze(-1), labels)
        return {'loss': loss, 'logits': evaluation_score}

class PromptEnhancedDataset(Dataset):
    """为评估定制的数据集类"""
    def __init__(self, src_texts, mt_texts, targets, tokenizer, is_training_set=False):
        self.src_texts = src_texts
        self.mt_texts = mt_texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.is_training_set = is_training_set # 对于评估，这应总是False

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = str(self.src_texts[idx])
        mt_text = str(self.mt_texts[idx])
        target = float(self.targets[idx])
        
        # for_training=self.is_training_set 确保评估时Prompt不包含答案
        evaluation_prompt = PromptEnhancedQwenRegression.create_evaluation_prompt(
            src_text, mt_text, target, for_training=self.is_training_set
        )
        
        encoding = self.tokenizer(
            evaluation_prompt,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.float32)
        }

def compute_metrics(predictions, labels):
    """计算回归评估指标"""
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    pearson_r, _ = pearsonr(predictions, labels)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r
    }

# ==============================================================================
#  3. MAIN EVALUATION LOGIC (更新后版本)
# ==============================================================================

if __name__ == "__main__":
    try:
        # --- 设置设备 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # --- 加载分词器 (从检查点加载，这通常是正确的) ---
        print(f"\n--- Loading Tokenizer ---")
        print(f"Path: {CHECKPOINT_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- 加载模型 (修正后的两步法) ---
        print(f"\n--- Loading Model ---")
        
        # 步骤 1: 使用【基础模型路径】来构建正确的模型“骨架”
        print(f"Building model structure from: {BASE_MODEL_PATH}")
        model = PromptEnhancedQwenRegression(BASE_MODEL_PATH)

        # 步骤 2: 从您的【检查点路径】加载微调后的权重到这个“骨架”中
        print(f"Loading fine-tuned weights from: {CHECKPOINT_PATH}")
        state_dict_path = os.path.join(CHECKPOINT_PATH, 'pytorch_model.bin')
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
        
        model.to(device)
        model.eval() # 设置为评估模式
        print("Model loaded successfully.")

        # ... (后续的数据加载、映射和评估循环代码保持不变) ...
        
        print(f"\n--- Loading & Processing Evaluation Data ---")
        print(f"Evaluation file: {EVAL_FILE_PATH}")
        print(f"Using training file for scale reference: {TRAIN_FILE_PATH}")
        df_eval = pd.read_csv(EVAL_FILE_PATH, sep='\t', encoding='utf-8')
        eval_src = df_eval['src'].astype(str).tolist()
        eval_mt = df_eval['mt'].astype(str).tolist()
        # ... (根据您的评估目标，这里可能是 'zmean' 或 'teacher_score')
        eval_targets_raw = df_eval['zmean'].astype(float).tolist()
        
        print("Applying the same score mapping as used in training...")
        new_min_target = -1.7
        new_max_target = 1.7
        df_train = pd.read_csv(TRAIN_FILE_PATH, sep='\t', usecols=['zmean'], encoding='latin-1')
        min_zmean_from_train = df_train['zmean'].min()
        max_zmean_from_train = df_train['zmean'].max()
        
        def map_score_range(score, old_min, old_max, new_min=new_min_target, new_max=new_max_target):
            if old_max == old_min: return new_min
            return new_min + (new_max - new_min) * (score - old_min) / (old_max - old_min)
            
        eval_targets_mapped = [map_score_range(s, old_min=min_zmean_from_train, old_max=max_zmean_from_train) for s in eval_targets_raw]
        print("Data mapping complete.")

        eval_dataset = PromptEnhancedDataset(
            eval_src, eval_mt, eval_targets_mapped, tokenizer, is_training_set=False
        )
        eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE)
        
        print(f"\n--- Running evaluation on {len(eval_dataset)} samples ---")
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                all_preds.extend(logits.squeeze(-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("\n--- Evaluation Finished ---")
        final_metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        pythonic_metrics = {key: float(value) for key, value in final_metrics.items()}

        print("\n" + "="*40)
        print("           FINAL EVALUATION RESULTS")
        for key, value in pythonic_metrics.items():
            print(f"  - {key.upper():<18}: {value:.6f}")
        print("="*40)
        print("\n--- Saving evaluation results to JSON ---")
        
        # 准备要保存的报告内容
        report_to_save = {
            "evaluation_data_path": EVAL_FILE_PATH,
            "evaluation_timestamp_utc": datetime.utcnow().isoformat(),
            "num_samples": len(all_labels),
            "metrics": pythonic_metrics
        }
        
        # 定义保存路径
        output_path = os.path.join(CHECKPOINT_PATH, "evaluation_results.json")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_to_save, f, indent=4)
            
        print(f"Results successfully saved to: {output_path}")

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        traceback.print_exc()
