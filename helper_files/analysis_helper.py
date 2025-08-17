import os
import pandas as pd
from pathlib import Path
import glob
import re
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def count_lines_in_file(filepath):
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except UnicodeDecodeError:
        # Try with another common encoding if UTF-8 fails
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return sum(1 for _ in f)
        except Exception as e:
            print(f"Error counting lines in {filepath}: {e}")
            return 0
    except Exception as e:
        print(f"Error counting lines in {filepath}: {e}")
        return 0

def analyze_language_pairs(base_dir: str) -> pd.DataFrame:
    """分析所有语言对的文件和行数"""
    # Find all TSV files in the directory
    tsv_files = glob.glob(os.path.join(base_dir, "*.tsv"))
    print(f"Found {len(tsv_files)} TSV files in {base_dir}")
    
    # Group files by language pair
    lang_pair_files = {}
    
    # 修改正则表达式模式，增加对括号格式的支持
    # 匹配以下模式:
    # - combined_en-de.tsv
    # - abnormal_en-de.tsv
    # - en-de.tsv
    # - en-de_smt.tsv  
    # - en-de(smt).tsv  (新增支持)
    lang_pair_pattern = r'((?:combined_)?(?:abnormal_)?([a-z]{2,4}\-[a-z]{2,4})(?:_[a-z]+|\([a-z]+\))?\.tsv)'
    
    for tsv_file in tsv_files:
        filename = os.path.basename(tsv_file)
        match = re.match(lang_pair_pattern, filename)
        
        if match:
            full_match, lang_pair = match.groups()
            if lang_pair not in lang_pair_files:
                lang_pair_files[lang_pair] = []
            lang_pair_files[lang_pair].append(tsv_file)
        else:
            # 尝试使用更宽松的模式再次匹配
            looser_match = re.search(r'([a-z]{2,4}\-[a-z]{2,4})', filename)
            if looser_match:
                lang_pair = looser_match.group(1)
                if lang_pair not in lang_pair_files:
                    lang_pair_files[lang_pair] = []
                lang_pair_files[lang_pair].append(tsv_file)
                print(f"  Using looser match for file: {filename} -> {lang_pair}")
            else:
                print(f"  Skipping file with no language pair match: {filename}")
    
    results = []
    
    for lang_pair in sorted(lang_pair_files.keys()):
        print(f"Processing LP: {lang_pair}")
        
        tsv_files = lang_pair_files[lang_pair]
        total_lines = 0
        file_details = []
        
        for tsv_file in tsv_files:
            filename = os.path.basename(tsv_file)
            try:
                lines = count_lines_in_file(tsv_file)
                total_lines += lines
                file_details.append({
                    'filename': filename,
                    'lines': lines,
                    'size_mb': os.path.getsize(tsv_file) / (1024 * 1024)
                })
                print(f"  - {filename}: {lines:,} lines")
            except Exception as e:
                print(f"  - Error processing {filename}: {str(e)}")
                file_details.append({
                    'filename': filename,
                    'lines': 0,
                    'size_mb': os.path.getsize(tsv_file) / (1024 * 1024),
                    'error': str(e)
                })
        
        results.append({
            'language_pair': lang_pair,
            'num_files': len(tsv_files),
            'total_lines': total_lines,
            'files': file_details
        })
        print(f"  In total: {total_lines:,} lines\n")
    
    # If no language pairs were found, check if files have a different naming pattern
    if not results:
        print("No language pair files found with standard naming. Analyzing all TSV files:")
        
        total_lines = 0
        file_details = []
        
        for tsv_file in tsv_files:
            filename = os.path.basename(tsv_file)
            try:
                lines = count_lines_in_file(tsv_file)
                total_lines += lines
                file_details.append({
                    'filename': filename,
                    'lines': lines,
                    'size_mb': os.path.getsize(tsv_file) / (1024 * 1024)
                })
                print(f"  - {filename}: {lines:,} lines")
            except Exception as e:
                print(f"  - Error processing {filename}: {str(e)}")
                file_details.append({
                    'filename': filename,
                    'lines': 0,
                    'size_mb': os.path.getsize(tsv_file) / (1024 * 1024),
                    'error': str(e)
                })
        
        results.append({
            'language_pair': 'unknown',
            'num_files': len(tsv_files),
            'total_lines': total_lines,
            'files': file_details
        })
        print(f"  In total: {total_lines:,} lines\n")
    
    # Convert results to DataFrame before returning
    return pd.DataFrame(results)

def analyze_and_visualize_tsv_data(data_dir, output_csv_name=None):
    """
    Analyze and visualize TSV data from language pair files.
    
    Args:
        data_dir (str): Directory containing the language pair TSV files
        output_csv_name (str, optional): Name for the detailed report CSV file.
                                       Defaults to 'language_pair_detailed_report.csv'
    
    Returns:
        tuple: (summary_table, df_detailed) - DataFrame summaries of the analysis
    """
    
    # If no output name is provided, create a default based on the directory name
    if output_csv_name is None:
        output_csv_name = f"{os.path.basename(data_dir)}_detailed_report.csv"
    
    # Analyze language pairs
    df_summary = analyze_language_pairs(data_dir)
    
    # Create summary table
    summary_table = df_summary[['language_pair', 'num_files', 'total_lines']].copy()
    summary_table = summary_table.sort_values('total_lines', ascending=False)
    
    # Add formatted lines column
    summary_table['formatted_lines'] = summary_table['total_lines'].apply(lambda x: f"{x:,}")
    
    # Calculate totals
    total_files = summary_table['num_files'].sum()
    total_lines = summary_table['total_lines'].sum()
    
    # Add total row
    total_row = pd.DataFrame({
        'language_pair': ['Total'],
        'num_files': [total_files],
        'total_lines': [total_lines],
        'formatted_lines': [f"{total_lines:,}"]
    })
    summary_table = pd.concat([summary_table, total_row], ignore_index=True)
    
    # Display summary table
    print(f"\n=== {os.path.basename(data_dir)} STATS ===\n")
    display(summary_table[['language_pair', 'num_files', 'formatted_lines']])
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Filter out the "Total" row for plotting
    data_for_plot = summary_table[summary_table['language_pair'] != 'Total'].copy()
    
    # Bar chart - lines per language pair
    plt.subplot(2, 1, 1)
    plt.bar(data_for_plot['language_pair'], data_for_plot['total_lines'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Num of lines')
    plt.title('Total lines for all LPs')
    plt.tight_layout()
    
    # Pie chart - line distribution
    plt.subplot(2, 1, 2)
    plt.pie(data_for_plot['total_lines'], 
            labels=data_for_plot['language_pair'],
            autopct='%1.1f%%',
            startangle=90)
    plt.title('Percentage of lines of all LPs')
    plt.tight_layout()
    plt.show()
    
    # Display detailed file information
    print("\n=== Detailed Info ===\n")
    for idx, row in df_summary.iterrows():
        print(f"\n{row['language_pair']}:")
        for file_info in row['files']:
            print(f"  - {file_info['filename']}: {file_info['lines']:,} lines ({file_info['size_mb']:.2f} MB)")
    
    # Create detailed report
    detailed_report = []
    for idx, row in df_summary.iterrows():
        for file_info in row['files']:
            detailed_report.append({
                'language_pair': row['language_pair'],
                'filename': file_info['filename'],
                'lines': file_info['lines'],
                'size_mb': file_info['size_mb']
            })
    df_detailed = pd.DataFrame(detailed_report)
    
    # Save detailed report
    df_detailed.to_csv(output_csv_name, index=False)
    print(f"\nReport saved to: {output_csv_name}")
    
    # Generate statistics
    print("\n=== Stats ===")
    print(f"Total number of LPs: {len(data_for_plot)}")
    print(f"Total number of files: {total_files}")
    print(f"Total lines: {total_lines:,}")
    
    if len(data_for_plot) > 0:
        print(f"Average lines per LP: {total_lines // len(data_for_plot):,}")
        
        # Sort to get largest and smallest language pairs
        sorted_data = data_for_plot.sort_values('total_lines', ascending=False)
        print(f"Largest LP: {sorted_data.iloc[0]['language_pair']} ({sorted_data.iloc[0]['total_lines']:,} lines)")
        print(f"Smallest LP: {sorted_data.iloc[-1]['language_pair']} ({sorted_data.iloc[-1]['total_lines']:,} lines)")
    
    return summary_table, df_detailed