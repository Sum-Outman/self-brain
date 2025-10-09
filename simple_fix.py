# -*- coding: utf-8 -*-
"""
Simple script to fix triple quote mismatches in initialize_system.py
"""

import sys
import os


def fix_triple_quotes(file_path, output_path):
    """Fix triple quote mismatches in the file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Checking {file_path} for triple quote mismatches...")
        
        # Count triple double quotes
        triple_double = content.count('"""')
        print(f"Found {triple_double} triple double quotes")
        
        # Count triple single quotes
        triple_single = content.count("'''")
        print(f"Found {triple_single} triple single quotes")
        
        # Create a copy of the content to work with
        fixed_content = content
        
        # Simple approach: Replace all triple quotes with single quotes
        # This is a brute-force approach but should fix the syntax errors
        fixed_content = fixed_content.replace('"""', '"')
        fixed_content = fixed_content.replace("''", "'")  # Fix any leftover pairs from replacing '''
        
        # Fix the first line (encoding declaration)
        if fixed_content.startswith('# -*- coding: utf-8 -*-'):
            print("Preserving encoding declaration")
        else:
            # Add encoding declaration if missing
            fixed_content = '# -*- coding: utf-8 -*-\n' + fixed_content
        
        # Write the fixed content to a new file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Fixed file created: {output_path}")
        print("This is a simplified fix - we've replaced all triple quotes with single quotes.")
        print("Please test this file and may need further manual adjustments.")
        
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python simple_fix.py <input_file_path> <output_file_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    fix_triple_quotes(input_path, output_path)