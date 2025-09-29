# -*- coding: utf-8 -*-
"""
This script checks for matching triple quotes in a Python file.
"""

import sys
import os


def check_triple_quotes(file_path):
    """Check for matching triple quotes in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check triple double quotes
        double_quotes = content.count('"""')
        if double_quotes % 2 != 0:
            print(f"Warning: Uneven number of triple double quotes: {double_quotes}")
            
        # Check triple single quotes
        single_quotes = content.count("'''")
        if single_quotes % 2 != 0:
            print(f"Warning: Uneven number of triple single quotes: {single_quotes}")
            
        # Try to compile the file to find exact error location
        try:
            compile(content, file_path, 'exec')
            print("File compiled successfully! No syntax errors detected.")
        except SyntaxError as e:
            print(f"SyntaxError: {e.msg} at line {e.lineno}")
            
    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_quotes.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    check_triple_quotes(file_path)