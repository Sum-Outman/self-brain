# -*- coding: utf-8 -*-
"""
This script uses Python's tokenize module to tokenize a Python file and find where
triple quotes might be unterminated.
"""

import sys
import os
import tokenize
from io import StringIO


def check_triple_quotes_tokens(file_path):
    """Check triple quotes using Python's tokenize module."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use tokenize to analyze the file
        token_stream = tokenize.generate_tokens(StringIO(content).readline)
        
        triple_double_quotes_count = 0
        triple_single_quotes_count = 0
        
        try:
            for token_type, token_string, start, end, line_text in token_stream:
                if token_type == tokenize.STRING:
                    # Check for triple quotes
                    if token_string.startswith('"""') or token_string.endswith('"""'):
                        triple_double_quotes_count += 1
                        print(f"Triple double quotes at line {start[0]}, col {start[1]}")
                    elif token_string.startswith("'''") or token_string.endswith("'''"):
                        triple_single_quotes_count += 1
                        print(f"Triple single quotes at line {start[0]}, col {start[1]}")
        except tokenize.TokenError as e:
            print(f"TokenError: {e}")
        except Exception as e:
            print(f"Error during tokenization: {e}")
        
        print(f"\nTotal triple double quotes: {triple_double_quotes_count}")
        print(f"Total triple single quotes: {triple_single_quotes_count}")
        
        if triple_double_quotes_count % 2 != 0:
            print("Warning: Uneven number of triple double quotes!")
        if triple_single_quotes_count % 2 != 0:
            print("Warning: Uneven number of triple single quotes!")
        
    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_tokens.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    check_triple_quotes_tokens(file_path)