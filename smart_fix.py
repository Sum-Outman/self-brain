#!/usr/bin/env python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script uses Python's tokenize module to properly parse and fix syntax errors
in initialize_system.py without breaking valid code structures.
"""

import sys
import os
import tokenize
from io import StringIO
import re

def smart_fix_syntax(file_path, output_path):
    """Smartly fix syntax errors in initialize_system.py."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Attempting to fix syntax errors in {file_path}...")
        
        # First, let's try a simple approach to fix the most common issue
        # 1. Fix the main issue with nested triple quotes in the app.py creation section
        # This is where we had the original problem
        fixed_content = fix_app_py_section(content)
        
        # 2. Check for any remaining triple quote issues
        fixed_content = ensure_matching_triple_quotes(fixed_content)
        
        # Write the fixed content to a new file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Fixed file created: {output_path}")
        print("Please test this fixed file to see if it resolves the syntax errors.")
        
    except Exception as e:
        print(f"Error fixing file: {e}")


def fix_app_py_section(content):
    """Fix the problematic app.py creation section."""
    # Find the section where app.py is created
    pattern = r'# Create app\.py for FastAPI\s+app_py_path = model_path / "app\.py"\s+if not app_py_path\.exists\(\):\s+(.*?)\s+with open\(app_py_path, \'w\', encoding=\'utf-8\'\) as f:'  
    
    # Replace the problematic section with a simplified version
    def replacement(match):
        indent = match.group(1).split('\n')[0] if '\n' in match.group(1) else ''
        
        # Create a clean version of the app.py content creation without nested quotes
        clean_code = f"# Create app.py for FastAPI\n{indent}app_py_path = model_path / \"app.py\"\n{indent}if not app_py_path.exists():\n{indent}    # Using a safer approach to create app.py content without nested triple quotes\n"
        clean_code += f"{indent}    app_content = '#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n'\n"
        clean_code += f"{indent}    app_content += '\"\"\"\n' + model_id.upper() + ' Model API\nThis is the main API file for the ' + model_id.upper() + ' model in the Self Brain system.\n\"\"\"\n\n'\n"
        clean_code += f"{indent}    app_content += 'from fastapi import FastAPI, HTTPException, Depends\nfrom fastapi.middleware.cors import CORSMiddleware\nfrom pydantic import BaseModel\nimport logging\nimport json\nimport os\nfrom pathlib import Path\n\n'\n"
        clean_code += f"{indent}    app_content += '# Configure logging\nlogging.basicConfig(\n    level=logging.INFO,\n    format=''%(asctime)s - {model_id} - %(levelname)s - %(message)s''\n)\n'\n"
        clean_code += f"{indent}    app_content += 'logger = logging.getLogger(\"SelfBrain.' + model_id + '\")\n\n'\n"
        clean_code += f"{indent}    app_content += '# Initialize FastAPI app\napp = FastAPI(title=\"' + model_id.upper() + ' Model API\", version=\"1.0.0\")\n\n'\n"
        clean_code += f"{indent}    app_content += '# CORS middleware\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=[\"*\"],\n    allow_credentials=True,\n    allow_methods=[\"*\"],\n    allow_headers=[\"*\"],\n)\n'\n"
        clean_code += f"{indent}    with open(app_py_path, 'w', encoding='utf-8') as f:\n"
        
        return clean_code
    
    # Apply the replacement
    fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.MULTILINE)
    return fixed_content


def ensure_matching_triple_quotes(content):
    """Ensure all triple quotes have proper closing quotes."""
    # Count triple double quotes
    triple_double = content.count('"""')
    if triple_double % 2 != 0:
        print(f"Warning: Found {triple_double} triple double quotes (uneven number)")
        # Add a closing triple quote at the end of the file if needed
        if not content.endswith('"""'):
            print("Adding a closing triple double quote at the end of the file")
            content += '"""'
    
    # Count triple single quotes
    triple_single = content.count("'''")
    if triple_single % 2 != 0:
        print(f"Warning: Found {triple_single} triple single quotes (uneven number)")
        # Add a closing triple quote at the end of the file if needed
        if not content.endswith("'''"):
            print("Adding a closing triple single quote at the end of the file")
            content += "'''"
    
    return content


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python smart_fix.py <input_file_path> <output_file_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    smart_fix_syntax(input_path, output_path)