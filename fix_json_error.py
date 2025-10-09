# Script to fix JSON formatting error in KProgrammingModelTrainer

import re

# Read the file content
file_path = r'd:\shiyan\web_interface\training_manager\model_trainer.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the JSON formatting error in suggestions array
# Replace the problematic line with the corrected version
fixed_content = re.sub(
    r'"suggestions": \["Add a colon after \'range\(10\)\'"\s*",',
    '"suggestions": ["Add a colon after \'range\(10\)\'"],',
    content,
    flags=re.MULTILINE
)

# Save the fixed content back to the file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("JSON formatting error fixed successfully!")