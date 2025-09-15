#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Programming Model - Capable of autonomous programming and system improvement

import os
import ast
import json
import subprocess
import sys
import logging
import time
import threading
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import difflib
import traceback
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('programming_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ProgrammingModel:
    """Programming Model - Capable of autonomous programming and system improvement"""
    
    def __init__(self, knowledge_base=None, language='en'):
        self.language = language
        self.knowledge_base = knowledge_base
        self.codebase_map = {}
        self.project_dependencies = {}
        self.code_history = []
        self.last_self_improvement = 0
        self.self_improvement_interval = 3600
        self.model_environment_map = {}
        self.supported_languages = ['python', 'javascript', 'java', 'c++', 'go']
        self.test_frameworks = {
            'python': 'pytest',
            'javascript': 'jest',
            'java': 'junit',
            'c++': 'gtest',
            'go': 'testing'
        }
        
        # Initialize codebase map
        self._build_codebase_map()
        self._analyze_project_dependencies()
        self._start_self_improvement_monitor()
    
    def _build_codebase_map(self):
        """Build codebase map"""
        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            models_dir = project_root / 'sub_models'
            
            if models_dir.exists():
                for model_dir in models_dir.iterdir():
                    if model_dir.is_dir():
                        model_id = model_dir.name.split('_')[1].upper() if '_' in model_dir.name else model_dir.name.upper()
                        self.model_environment_map[model_id] = {
                            'path': str(model_dir),
                            'main_file': str(model_dir / 'app.py') if (model_dir / 'app.py').exists() else None,
                            'dependencies': [],
                            'config_files': []
                        }
            
            logger.info("Codebase map construction completed")
            return True
        except Exception as e:
            logger.error(f"Error building codebase map: {str(e)}")
            return False
    
    def _analyze_project_dependencies(self):
        """Analyze project dependencies"""
        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            req_file = project_root / 'requirements.txt'
            
            if req_file.exists():
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            dep = line.split('==')[0].strip()
                            self.project_dependencies[dep] = line
            
            logger.info("Project dependency analysis completed")
            return True
        except Exception as e:
            logger.error(f"Error analyzing project dependencies: {str(e)}")
            return False
    
    def _start_self_improvement_monitor(self):
        """Start self-improvement monitoring thread"""
        def monitor():
            while True:
                current_time = time.time()
                if current_time - self.last_self_improvement > self.self_improvement_interval:
                    self._perform_self_improvement()
                    self.last_self_improvement = current_time
                time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info("Self-improvement monitoring thread started")
    
    def _perform_self_improvement(self):
        """Perform self-improvement"""
        try:
            logger.info("Starting self-improvement process...")
            # Analyze codebase for improvements
            improvements = self._analyze_code_improvements()
            
            if improvements:
                for improvement in improvements:
                    self._apply_improvement(improvement)
                
                logger.info(f"Applied {len(improvements)} self-improvements")
            else:
                logger.info("No improvements needed at this time")
                
        except Exception as e:
            logger.error(f"Self-improvement error: {str(e)}")
    
    def _analyze_code_improvements(self):
        """Analyze code for potential improvements"""
        improvements = []
        
        # Check for missing error handling
        for model_id, model_info in self.model_environment_map.items():
            if model_info['main_file'] and os.path.exists(model_info['main_file']):
                with open(model_info['main_file'], 'r') as f:
                    content = f.read()
                    if 'try:' not in content and 'except' not in content:
                        improvements.append({
                            'type': 'add_error_handling',
                            'file': model_info['main_file'],
                            'model_id': model_id
                        })
        
        return improvements
    
    def _apply_improvement(self, improvement):
        """Apply specific improvement"""
        try:
            if improvement['type'] == 'add_error_handling':
                self._add_error_handling(improvement['file'])
        except Exception as e:
            logger.error(f"Error applying improvement: {str(e)}")
    
    def _add_error_handling(self, file_path):
        """Add error handling to file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Simple error handling wrapper
            if 'def main():' in content:
                new_content = content.replace('def main():', 'def main():\n    try:')
                new_content += '\n    except Exception as e:\n        logger.error(f"Error in main: {e}")\n        return None'
                
                with open(file_path, 'w') as f:
                    f.write(new_content)
                
                logger.info(f"Added error handling to {file_path}")
        except Exception as e:
            logger.error(f"Error adding error handling: {str(e)}")
    
    def generate_code(self, prompt: str, language: str = 'python', context: Dict = None) -> Dict:
        """Generate code based on prompt"""
        try:
            if language not in self.supported_languages:
                return {'error': f'Unsupported language: {language}'}
            
            # Generate basic code structure
            code_template = self._get_code_template(language, prompt, context)
            
            return {
                'code': code_template,
                'language': language,
                'generated_at': time.time(),
                'context': context or {}
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_code_template(self, language: str, prompt: str, context: Dict) -> str:
        """Get code template for given language and prompt"""
        templates = {
            'python': '''def main():
    """Generated function based on: {prompt}"""
    # TODO: Implement based on requirements
    pass

if __name__ == "__main__":
    main()''',
            'javascript': '''function main() {
    // Generated function based on: {prompt}
    // TODO: Implement based on requirements
    return null;
}

if (require.main === module) {
    main();
}'''
        }
        
        template = templates.get(language, templates['python'])
        return template.format(prompt=prompt)
    
    def optimize_code(self, code: str, language: str = 'python') -> Dict:
        """Optimize existing code"""
        try:
            # Basic optimization suggestions
            suggestions = []
            
            # Check for common issues
            if 'import *' in code:
                suggestions.append("Avoid using 'import *' - use specific imports")
            
            if 'print(' in code and 'logging' not in code:
                suggestions.append("Consider using logging instead of print statements")
            
            # Basic formatting
            optimized_code = code.replace('\t', '    ')
            
            return {
                'original_code': code,
                'optimized_code': optimized_code,
                'suggestions': suggestions,
                'language': language
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_code(self, code: str, language: str = 'python') -> Dict:
        """Analyze code for issues and improvements"""
        try:
            issues = []
            
            # Basic syntax check
            if language == 'python':
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    issues.append({
                        'type': 'syntax_error',
                        'line': e.lineno,
                        'message': str(e)
                    })
            
            # Complexity analysis
            lines = code.split('\n')
            if len(lines) > 100:
                issues.append({
                    'type': 'complexity',
                    'message': 'Function is too long, consider breaking it down'
                })
            
            return {
                'issues': issues,
                'metrics': {
                    'lines_of_code': len(lines),
                    'complexity_score': min(100, len(lines))
                },
                'language': language
            }
        except Exception as e:
            return {'error': str(e)}

# Initialize programming model
programming_model = ProgrammingModel()

# API Routes
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'K_programming',
        'timestamp': time.time()
    })

@app.route('/api/generate_code', methods=['POST'])
def generate_code():
    """Generate code endpoint"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        language = data.get('language', 'python')
        context = data.get('context', {})
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        result = programming_model.generate_code(prompt, language, context)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize_code', methods=['POST'])
def optimize_code():
    """Optimize code endpoint"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        language = data.get('language', 'python')
        
        if not code:
            return jsonify({'error': 'Code is required'}), 400
        
        result = programming_model.optimize_code(code, language)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_code', methods=['POST'])
def analyze_code():
    """Analyze code endpoint"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        language = data.get('language', 'python')
        
        if not code:
            return jsonify({'error': 'Code is required'}), 400
        
        result = programming_model.analyze_code(code, language)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def get_models():
    """Get available models"""
    return jsonify({
        'models': list(programming_model.model_environment_map.keys()),
        'supported_languages': programming_model.supported_languages
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5011, debug=False)
