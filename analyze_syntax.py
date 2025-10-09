# -*- coding: utf-8 -*-
"""
This script uses Python's ast module to analyze syntax errors in a Python file.
"""

import sys
import os
import ast
import tokenize
from io import StringIO


def find_syntax_error(file_path):
    """Find and print detailed information about syntax errors in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the file with ast
        try:
            ast.parse(content, filename=file_path)
            print("File parsed successfully by ast module! No syntax errors detected.")
            return True
        except SyntaxError as e:
            print(f"SyntaxError: {e.msg}")
            print(f"  Line: {e.lineno}, Column: {e.offset}")
            print(f"  Text: {e.text}")
            
            # Print the surrounding lines
            lines = content.split('\n')
            start_line = max(1, e.lineno - 2)
            end_line = min(len(lines), e.lineno + 2)
            
            print("\nSurrounding lines:")
            for i in range(start_line-1, end_line):
                line_num = i + 1
                prefix = '>' if line_num == e.lineno else ' '
                print(f"{prefix}{line_num}: {lines[i]}")
            
            return False
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_syntax.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    find_syntax_error(file_path)