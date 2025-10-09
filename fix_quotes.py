# -*- coding: utf-8 -*-
"""
This script fixes the unterminated triple quotes in initialize_system.py
"""

import sys
import os


def fix_all_unterminated_quotes(file_path):
    """Fix all unterminated triple quotes in initialize_system.py."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count triple double quotes
        triple_double_quotes = content.count('"""')
        print(f"Found {triple_double_quotes} occurrences of triple double quotes")
        
        # Count triple single quotes
        triple_single_quotes = content.count("'''")
        print(f"Found {triple_single_quotes} occurrences of triple single quotes")
        
        # If odd number of triple quotes, replace them all with single quotes
        # This is a brute-force approach to fix the syntax error
        if triple_double_quotes % 2 != 0:
            print("Replacing all triple double quotes with single quotes...")
            content = content.replace('"""', "'")
        
        if triple_single_quotes % 2 != 0:
            print("Replacing all triple single quotes with single quotes...")
            content = content.replace("''", "'")  # Replace pairs of single quotes first
            content = content.replace("''", "'")  # Again to handle any remaining
        
        # Write the fixed content back to a new file for testing
        fixed_file_path = file_path + '.fixed'
        with open(fixed_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Fixed file created: {fixed_file_path}")
        print("Please test this fixed file to see if it resolves the syntax errors.")
        
    except Exception as e:
        print(f"Error fixing file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_quotes.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    fix_all_unterminated_quotes(file_path)