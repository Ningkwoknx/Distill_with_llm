import openai
import pandas as pd
import os
import time
from tqdm import tqdm
import argparse
import concurrent.futures

######MODEL DEFINITION#######
GPT_MODEL = "o4-mini"

#####API INITIATION#####
try:
    client = openai.OpenAI()
except openai.OpenAIError:
    print("Error: no API environment variable found.")
    exit()

SYSTEM_PROMPT = """
You are an annotator for the quality of machine translation. You have a pair of sentences and a score indicating the translation quality. Your task is to identify errors, assess the quality of the translation, and explain why the translation gets this score.
The score is in range [-2, 2], -2 signifies that the translation is at the worst quality, inhibiting comprehension of the text; 2 signifies that the translation is at the best quality, fluent and accurate, used proper terms. The scores in between indicates major or minor errors: Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but they do not disrupt the flow or hinder comprehension. The score's position within the -2 to 2 range indicates subtle, yet significant, differences in translation quality. Do not repeat the sentences or give any background knowledge that is irrelevant to MTQE, give only reasonings and address different aspects. **Your response must be direct and concise. Do not include any headers (like 'Assessment' or 'Reasoning'), bullet points, or concluding summary paragraphs (like 'Overall, ...'). Output only the core reasoning as a single, dense paragraph.**
"""
# This function is designed to be called by each thread. It processes one row of data.
def get_explanation_for_row(row_data):
    """
    A helper function that gets a GPT explanation for a single row of data.
    It takes a tuple (index, row_series) and returns a tuple (index, explanation_text).
    """
    index, row = row_data  # Unpack the data tuple
    src = row['src']
    mt = row['mt']
    score = row['zmean']
    user_prompt = f"Source: {src}\nTranslation: {mt}\nScore:{score}"

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=2000 # Increased token limit for safety
        )
        explanation = response.choices[0].message.content.strip()
        return (index, explanation) # Return the index and the result
    except Exception as e:
        # tqdm.write is thread-safe and won't mess up the progress bar
        tqdm.write(f"Error on index {index}: {e}")
        return (index, None) # Return None on failure to indicate an error occurred



def process_file(input_path, output_path):
    """
    Loads a file with auto-detected encoding, gets GPT explanations, and saves.
    Supports resuming from a checkpoint.
    """

    print(f"Loading file: {input_path}")
    
    try:
        # try utf-8 first
        df = pd.read_csv(input_path, sep='\t', encoding='utf-8')
        print("File loaded successfully with utf-8.")
    except UnicodeDecodeError:
        # try latin-1 if utf-8 doesn't work
        print("Failed to load with utf-8. Falling back to latin-1...")
        df = pd.read_csv(input_path, sep='\t', encoding='latin-1')
        print("File loaded successfully with latin-1.")

    if 'gpt_explanation' not in df.columns:
        df['gpt_explanation'] = pd.NA

    if os.path.exists(output_path):
        print(f"Found existing output file, loading and resuming...")
        df_output = pd.read_csv(output_path, sep='\t')
        df.update(df_output)
    
    # --- NEW: Identify which rows need to be processed ---
    rows_to_process = [row_tuple for row_tuple in df[df['gpt_explanation'].isna()].iterrows()]

    if not rows_to_process:
        print("All rows have already been processed. Task complete!")
        return

    print(f"Total rows: {len(df)}. Rows to process: {len(rows_to_process)}.")
    
    MAX_WORKERS = 120

    print(f"Starting parallel processing with {MAX_WORKERS} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # executor.map applies the 'get_explanation_for_row' function to each item in 'rows_to_process'.
        # It automatically manages the thread pool and collects results.
        results_iterator = executor.map(get_explanation_for_row, rows_to_process)

        # Wrap the results iterator with tqdm to create a live progress bar.
        save_interval = 200  # Save progress every 200 processed rows
        processed_in_batch = 0

        for result in tqdm(results_iterator, total=len(rows_to_process), desc="Processing data"):
            index, explanation = result
            if explanation is not None:
                df.at[index, 'gpt_explanation'] = explanation
            
            processed_in_batch += 1
            # --- NEW: Checkpoint saving logic ---
            if processed_in_batch % save_interval == 0:
                tqdm.write(f"\n--- Checkpoint: Saving progress ({processed_in_batch}/{len(rows_to_process)} processed) to {output_path}... ---")
                df.to_csv(output_path, sep='\t', index=False)
                tqdm.write("--- Save complete. Resuming... ---")
    print(f"\nFinished processing! Saving final file...")
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Task complete! File saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get GPT explanations. Run with arguments for a specific file, or without arguments for the hardcoded batch job."
    )
    parser.add_argument("-i", "--input_path", type=str, help="Path to the input TSV file.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the output TSV file.")
    
    args = parser.parse_args()

    if args.input_path and args.output_path:
        print("\n" + "="*60)
        print("Running in single-file mode...")
        print(f"  Input: {args.input_path}")
        print(f"  Output: {args.output_path}")
        print("="*60)
        process_file(args.input_path, args.output_path)

    elif args.input_path or args.output_path:
        print("Error: You must provide both --input_path (-i) and --output_path (-o).")
    
    else:
        print("\n" + "="*60)
        print("No command-line arguments detected. Running in hardcoded batch mode...")
        print("="*60)

        files_to_process = [
            {
                "name": "Training Set",
                "input": "../data/zmean_train_80.tsv",
                "output": "../data/zmean_train_80_with_gpt_reasonings.tsv"
            },
            {
                "name": "Development Set",
                "input": "../data/zmean_dev_10.tsv",
                "output": "../data/zmean_dev_10_with_gpt_reasonings.tsv"
            }
        ]

        for file_info in files_to_process:
            print("\n" + "~"*60)
            print(f"Starting to process: {file_info['name']}")
            print("~"*60)
            process_file(
                input_path=file_info['input'],
                output_path=file_info['output']
            )
            print(f"\n{file_info['name']} processing complete.")

    print("\n" + "="*60)
    print("All tasks finished!")
    print("="*60)
