#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved script to fix triple quote mismatches in initialize_system.py
This version handles multi-line docstrings properly.
"""

import sys
import os
import re

def fix_docstrings(file_path, output_path):
    """Fix multi-line docstrings in the file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Fixing docstrings in {file_path}...")
        
        # First, try to fix the document string at the beginning
        # Replace the problematic single-quoted multi-line string with proper triple quotes
        fixed_content = re.sub(r'"\\nSelf Brain System Initialization Script\\n.*?for training from scratch and all features are enabled\\.\\n"', 
                              '"""\nSelf Brain System Initialization Script\nThis script initializes the Self Brain AGI system, ensuring all models are properly configured\nfor training from scratch and all features are enabled.\n"""', 
                              content, 
                              flags=re.DOTALL)
        
        # Now let's fix other possible docstrings in the file
        # Look for function definitions followed by improperly formatted docstrings
        # Example pattern for function definitions followed by a problematic docstring
        fixed_content = re.sub(r'def\s+(\w+)\s*\(.*?\):\s*"\n(.*?)\n"', 
                              lambda m: f'def {m.group(1)}(...):\n    """\n{m.group(2)}\n    """', 
                              fixed_content, 
                              flags=re.DOTALL)
        
        # Also check for class definitions with problematic docstrings
        fixed_content = re.sub(r'class\s+(\w+)\s*\(.*?\):\s*"\n(.*?)\n"', 
                              lambda m: f'class {m.group(1)}(...):\n    """\n{m.group(2)}\n    """', 
                              fixed_content, 
                              flags=re.DOTALL)
        
        # Write the fixed content to a new file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Fixed file created: {output_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_docstrings.py <input_file_path> <output_file_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    fix_docstrings(input_path, output_path)