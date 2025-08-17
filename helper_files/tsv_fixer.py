#!/usr/bin/env python3
"""
TSV file format fixing script
Used to fix formatting issues in WMT datasets, ensuring each line has exactly two columns (src/mt)
"""

import argparse
import sys
from typing import List, Tuple


def process_tsv_file(input_file: str, output_file: str) -> Tuple[int, int]:
    """
    Process TSV file, fix formatting issues
    
    Cleaning logic:
    1. First check if any column in a line contains \n
    2. If there's \n, add content after \n as the first column of a new line
    3. Check if any line has more than two columns
       - If it has four columns, move the extra two columns to a new line as first and second columns
       - If it has three columns, check if it's because content from the second column with \n
         pushed to the third column. In this case, apply the \n logic to add new lines,
         then move the third column to the second column of the last added line
    4. Finally, check all lines to ensure each has exactly two columns with no \t or \n
    
    Args:
        input_file: Input TSV file path
        output_file: Output TSV file path
    
    Returns:
        (original line count, processed line count)
    """
    original_lines = 0
    processed_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            original_lines = len(lines)
        
        print(f"Original file lines: {original_lines}")
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip('\n')
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Split line to get columns
            columns = line.split('\t')
            column_count = len(columns)
            
            # If there are less than 1 column, skip
            if column_count < 1 or not line.strip():
                i += 1
                continue
            
            # Temporary storage for processed results
            result_rows = []
            
            # Phase 1: Handle single-column cases
            if column_count == 1:
                # Check if there's a tab character
                if '\t' in columns[0]:
                    # Re-split
                    parts = columns[0].split('\t', 1)
                    columns = [parts[0], parts[1] if len(parts) > 1 else ""]
                    column_count = len(columns)
                else:
                    # Only one column and no tab, add an empty column
                    columns.append("")
                    column_count = 2
            
            # Phase 2: Process newline characters \n
            # First add current line to results
            current_row = columns[:2]  # Ensure we only take the first two columns
            result_rows.append(current_row)
            
            # Check if first column has newlines
            if '\n' in columns[0] or '\r' in columns[0]:
                # Replace all \r with \n
                col_content = columns[0].replace('\r', '\n')
                # Split by \n
                parts = col_content.split('\n')
                # Update first column of current row
                result_rows[0][0] = parts[0]
                # Remaining parts go to first column of new rows
                for part in parts[1:]:
                    if part.strip():  # Ensure it's not an empty string
                        result_rows.append([part, ""])
            
            # Check if second column has newlines
            if column_count > 1 and ('\n' in columns[1] or '\r' in columns[1]):
                # Replace all \r with \n
                col_content = columns[1].replace('\r', '\n')
                # Split by \n
                parts = col_content.split('\n')
                # Update second column of current row
                result_rows[0][1] = parts[0]
                # Remaining parts go to first column of new rows
                for part in parts[1:]:
                    if part.strip():  # Ensure it's not an empty string
                        result_rows.append([part, ""])
            
            # Phase 3: Handle extra columns
            # Handle four-column case
            if column_count == 4:
                # Ensure current row has only two columns
                result_rows[0] = columns[:2]
                # Move last two columns to a new row
                result_rows.append([columns[2], columns[3]])
            
            # Handle three-column case
            elif column_count == 3:
                # Logic: If second column has \n, third column might be pushed content
                if '\n' in columns[1] or '\r' in columns[1]:
                    # Already processed second column newlines above, now handle third column
                    # Add third column to second column of the last added row
                    if len(result_rows) > 1:
                        result_rows[-1][1] = columns[2]
                    else:
                        # If no new rows were added, create one
                        result_rows.append(["", columns[2]])
                else:
                    # If second column has no \n, add third column as first column of new row
                    result_rows.append([columns[2], ""])
            
            # Final phase: Clean any \t in each column
            final_rows = []
            for row in result_rows:
                # Check if first column contains \t
                if '\t' in row[0]:
                    parts = row[0].split('\t', 1)
                    row[0] = parts[0]
                    # If second column is empty, use split part as second column
                    if not row[1]:
                        row[1] = parts[1] if len(parts) > 1 else ""
                    else:
                        # If second column is not empty, create a new row
                        if len(parts) > 1 and parts[1].strip():
                            final_rows.append([parts[1], ""])
                
                # Check if second column contains \t
                if '\t' in row[1]:
                    parts = row[1].split('\t')
                    row[1] = parts[0]
                    # Subsequent parts become new rows
                    for j in range(1, len(parts), 2):
                        col1 = parts[j] if j < len(parts) else ""
                        col2 = parts[j+1] if j+1 < len(parts) else ""
                        if col1.strip() or col2.strip():
                            final_rows.append([col1, col2])
                
                # Add processed row
                if row[0].strip() or row[1].strip():
                    final_rows.append(row)
            
            # Add processed rows to final results
            for row in final_rows:
                processed_lines.append(f"{row[0]}\t{row[1]}")
            
            i += 1
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in processed_lines:
                f.write(line + '\n')
        
        final_lines = len(processed_lines)
        print(f"Processed file lines: {final_lines}")
        
        return original_lines, final_lines
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Fix TSV file format issues')
    parser.add_argument('input_file', help='Input TSV file path')
    parser.add_argument('output_file', help='Output TSV file path')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed processing information')
    
    args = parser.parse_args()
    
    print(f"Starting to process file: {args.input_file}")
    original_lines, final_lines = process_tsv_file(args.input_file, args.output_file)
    
    print(f"\nProcessing complete!")
    print(f"Original lines: {original_lines}")
    print(f"Final lines: {final_lines}")
    print(f"Line change: {final_lines - original_lines}")
    print(f"Output file: {args.output_file}")


if __name__ == "__main__":
    main()
