# Copyright 2025 The AI Management System Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Programming Model API Service

from flask import Flask, jsonify, request
import threading
import time
import random
from datetime import datetime
import json
import os
import re
import hashlib
import ast
import traceback
from collections import defaultdict

app = Flask(__name__)

# Supported programming languages
SUPPORTED_LANGUAGES = [
    "python", "javascript", "java", "c++", "c#", "go", "rust", "typescript"
]

# Programming model status
programming_status = {
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "code_snippets": 0,
    "code_generated": 0,
    "code_analyzed": 0,
    "code_optimized": 0,
    "code_fixed": 0,
    "self_improvements": 0,  # New self-improvement counter
    "current_version": "1.0.0",
    "languages_supported": SUPPORTED_LANGUAGES
}

# Code knowledge base storage
CODE_KNOWLEDGE_FILE = "code_knowledge.json"
code_knowledge = defaultdict(dict)  # Store code snippets by language

def load_code_knowledge():
    """Load code knowledge base"""
    global code_knowledge
    if os.path.exists(CODE_KNOWLEDGE_FILE):
        try:
            with open(CODE_KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                code_knowledge = json.load(f)
            print("Code knowledge base loaded")
            # Update code snippet count
            count = 0
            for language in code_knowledge:
                count += len(code_knowledge[language])
            programming_status["code_snippets"] = count
        except Exception as e:
            print(f"Failed to load code knowledge base: {str(e)}")
    else:
        print("Using empty code knowledge base")

def save_code_knowledge():
    """Save code knowledge base to file"""
    try:
        with open(CODE_KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(code_knowledge, f, ensure_ascii=False, indent=2)
        print("Code knowledge base saved")
    except Exception as e:
        print(f"Failed to save code knowledge base: {str(e)}")

def programming_maintenance():
    """Regular maintenance of programming model"""
    while True:
        # Automatically save every 30 minutes
        save_code_knowledge()
        time.sleep(1800)

def generate_code_id(language, functionality):
    """Generate code snippet ID"""
    return hashlib.md5(f"{language}_{functionality}".encode('utf-8')).hexdigest()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "code_snippets": programming_status["code_snippets"],
        "version": programming_status["current_version"]
    })

@app.route('/get_status', methods=['GET'])
def get_status():
    """Get programming model status"""
    return jsonify({
        "status": "success",
        "status_info": {
            "last_updated": programming_status["last_updated"],
            "code_snippets": programming_status["code_snippets"],
            "code_generated": programming_status["code_generated"],
            "code_analyzed": programming_status["code_analyzed"],
            "code_optimized": programming_status["code_optimized"],
            "code_fixed": programming_status["code_fixed"],
            "current_version": programming_status["current_version"],
            "languages_supported": programming_status["languages_supported"]
        }
    })

@app.route('/generate_code', methods=['POST'])
def generate_code():
    """Generate code based on requirement description"""
    data = request.json
    if not data or 'description' not in data or 'language' not in data:
        return jsonify({"status": "error", "message": "Missing description or language parameter"}), 400
    
    description = data['description']
    language = data['language']
    context = data.get('context', "")
    
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"status": "error", "message": f"Unsupported language: {language}"}), 400
    
    programming_status["code_generated"] += 1
    
    # Simulate code generation
    code_snippet = ""
    if language == "python":
        code_snippet = f"# {description}\ndef solution():\n    # Implementation code\n    pass"
    elif language == "javascript":
        code_snippet = f"// {description}\nfunction solution() {{\n    // Implementation code\n}}"
    elif language == "java":
        code_snippet = f"// {description}\npublic class Solution {{\n    public static void main(String[] args) {{\n        // Implementation code\n    }}\n}}"
    elif language == "c++":
        code_snippet = f"// {description}\n#include <iostream>\n\nint main() {{\n    // Implementation code\n    return 0;\n}}"
    
    # Add to knowledge base
    snippet_id = generate_code_id(language, description)
    if language not in code_knowledge:
        code_knowledge[language] = {}
    
    code_knowledge[language][snippet_id] = {
        "id": snippet_id,
        "language": language,
        "description": description,
        "code": code_snippet,
        "context": context,
        "created_at": datetime.now().isoformat()
    }
    programming_status["code_snippets"] += 1
    
    return jsonify({
        "status": "success",
        "language": language,
        "code": code_snippet
    })

@app.route('/analyze_code', methods=['POST'])
def analyze_code():
    """Analyze code and provide improvement suggestions"""
    data = request.json
    if not data or 'code' not in data or 'language' not in data:
        return jsonify({"status": "error", "message": "Missing code or language parameter"}), 400
    
    code = data['code']
    language = data['language']
    
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"status": "error", "message": f"Unsupported language: {language}"}), 400
    
    programming_status["code_analyzed"] += 1
    
    # Simulate code analysis
    analysis_result = {
        "complexity": random.choice(["low", "medium", "high"]),
        "readability": random.choice(["good", "fair", "poor"]),
        "efficiency": random.choice(["efficient", "moderate", "inefficient"]),
        "security": random.choice(["secure", "vulnerable"]),
        "suggestions": [
            "Suggestion 1: Add more comments",
            "Suggestion 2: Optimize loop structures",
            "Suggestion 3: Use more efficient data structures"
        ],
        "potential_bugs": [
            "Potential issue 1: Unhandled boundary conditions",
            "Potential issue 2: Possible memory leaks"
        ]
    }
    
    # If it's Python, try AST analysis
    if language == "python":
        try:
            ast.parse(code)
            analysis_result["syntax"] = "valid"
        except SyntaxError as e:
            analysis_result["syntax"] = "invalid"
            analysis_result["error"] = str(e)
            analysis_result["line"] = e.lineno
            analysis_result["offset"] = e.offset
    
    return jsonify({
        "status": "success",
        "analysis_result": analysis_result
    })

@app.route('/optimize_code', methods=['POST'])
def optimize_code():
    """Optimize given code"""
    data = request.json
    if not data or 'code' not in data or 'language' not in data:
        return jsonify({"status": "error", "message": "Missing code or language parameter"}), 400
    
    code = data['code']
    language = data['language']
    
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"status": "error", "message": f"Unsupported language: {language}"}), 400
    
    programming_status["code_optimized"] += 1
    
    # Simulate code optimization
    optimized_code = code
    if language == "python":
        # Simple optimization examples
        optimized_code = code.replace("for i in range(len(list)):", "for item in list:")
        optimized_code = optimized_code.replace("if x == True:", "if x:")
    
    return jsonify({
        "status": "success",
        "original_code": code,
        "optimized_code": optimized_code,
        "optimization_details": "Simplified loops and conditional expressions"
    })

@app.route('/fix_code', methods=['POST'])
def fix_code():
    """Fix errors in code"""
    data = request.json
    if not data or 'code' not in data or 'language' not in data:
        return jsonify({"status": "error", "message": "Missing code or language parameter"}), 400
    
    code = data['code']
    language = data['language']
    error_message = data.get('error', "")
    
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"status": "error", "message": f"Unsupported language: {language}"}), 400
    
    programming_status["code_fixed"] += 1
    
    # Simulate code fixing
    fixed_code = code
    fix_details = []
    
    # Common error fixes
    if "SyntaxError" in error_message:
        if "missing parentheses" in error_message:
            fixed_code = code.replace("print 'hello'", "print('hello')")
            fix_details.append("Fixed print statement with missing parentheses")
    
    if "NameError" in error_message:
        if "undefined variable" in error_message or "is not defined" in error_message:
            match = re.search(r"name '(\w+)'", error_message)
            if match:
                var_name = match.group(1)
                fixed_code = f"{var_name} = None  # Initialize variable\n\n" + code
                fix_details.append(f"Initialized undefined variable: {var_name}")
            else:
                # Try more general pattern
                match = re.search(r"name '([^']+)'", error_message)
                if match:
                    var_name = match.group(1)
                    fixed_code = f"{var_name} = None  # Initialize variable\n\n" + code
                    fix_details.append(f"Initialized undefined variable: {var_name}")
                else:
                    fixed_code = "# Error fix: undefined variable\n" + code
                    fix_details.append("Added generic fix for undefined variable")
    
    if not fix_details:
        fixed_code = "# Error fix\n" + code
        fix_details.append("Added generic error handling")
    
    return jsonify({
        "status": "success",
        "original_code": code,
        "fixed_code": fixed_code,
        "fix_details": fix_details
    })

@app.route('/self_learn', methods=['POST'])
def self_learn():
    """Programming model self-learning"""
    data = request.json
    language = data.get('language', 'python')
    functionality = data.get('functionality', 'general programming')
    
    # Simulate self-learning process
    learned_snippets = []
    for i in range(3):  # Generate 3 new code snippets
        snippet_desc = f"{functionality} example {i+1}"
        snippet_id = generate_code_id(language, snippet_desc)
        
        if language not in code_knowledge:
            code_knowledge[language] = {}
        
        # Generate example code
        if language == "python":
            code = f"# {snippet_desc}\ndef example_{i+1}():\n    # Implementation code\n    pass"
        elif language == "javascript":
            code = f"// {snippet_desc}\nfunction example_{i+1}() {{\n    // Implementation code\n}}"
        else:
            code = f"// {snippet_desc}\n// Implementation code"
        
        code_knowledge[language][snippet_id] = {
            "id": snippet_id,
            "language": language,
            "description": snippet_desc,
            "code": code,
            "created_at": datetime.now().isoformat()
        }
        programming_status["code_snippets"] += 1
        learned_snippets.append(snippet_desc)
    
    return jsonify({
        "status": "success",
        "learned_snippets": learned_snippets,
        "new_snippets": len(learned_snippets)
    })

@app.route('/self_improve', methods=['POST'])
def self_improve():
    """Programming model self-improvement"""
    data = request.json
    improvement_target = data.get('target', 'general')
    
    # Simulate self-improvement process
    improvements = []
    
    # 1. Analyze current code knowledge base
    analysis_result = {
        "total_snippets": programming_status["code_snippets"],
        "languages_covered": list(code_knowledge.keys()),
        "coverage_score": random.randint(60, 95)
    }
    
    # 2. Identify improvement areas
    if improvement_target == 'performance':
        improvements.append("Optimize code execution efficiency")
        improvements.append("Reduce memory usage")
    elif improvement_target == 'security':
        improvements.append("Enhance input validation")
        improvements.append("Add security protection mechanisms")
    else:  # general improvement
        improvements.append("Refactor redundant code")
        improvements.append("Optimize API response structure")
        improvements.append("Enhance error handling mechanism")
    
    # 3. Apply improvements (simulated)
    programming_status["current_version"] = f"1.{programming_status['self_improvements'] + 1}.0"
    programming_status["self_improvements"] += 1
    
    # 4. Update knowledge base (simulated)
    new_snippets = []
    for imp in improvements:
        snippet_id = generate_code_id("python", f"self_improve_{imp}")
        code = f"# Self-improvement: {imp}\n# Implementation code..."
        
        if "python" not in code_knowledge:
            code_knowledge["python"] = {}
        
        code_knowledge["python"][snippet_id] = {
            "id": snippet_id,
            "language": "python",
            "description": f"Self-improvement: {imp}",
            "code": code,
            "created_at": datetime.now().isoformat()
        }
        programming_status["code_snippets"] += 1
        new_snippets.append(imp)
    
    return jsonify({
        "status": "success",
        "analysis": analysis_result,
        "improvements_applied": improvements,
        "new_snippets_added": new_snippets,
        "new_version": programming_status["current_version"],
        "total_improvements": programming_status["self_improvements"]
    })

if __name__ == '__main__':
    # Load code knowledge base
    load_code_knowledge()
    
    # Start programming model maintenance thread
    maintenance_thread = threading.Thread(target=programming_maintenance, daemon=True)
    maintenance_thread.start()
    
    # Start API service
    import sys
    port = int(os.environ.get('PORT', 5010))
    for arg in sys.argv:
        if arg.startswith('--port='):
            port = int(arg.split('=')[1])
        elif arg == '--port' and len(sys.argv) > sys.argv.index(arg) + 1:
            port = int(sys.argv[sys.argv.index(arg) + 1])
    app.run(host='0.0.0.0', port=port)
