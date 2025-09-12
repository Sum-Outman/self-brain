# -*- coding: utf-8 -*-
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0

"""
Unified Programming Model
Integrates standard and enhanced mode functionality
"""

import ast
import astroid
import subprocess
import tempfile
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
import threading
from dataclasses import dataclass
import re

@dataclass
class CodeAnalysis:
    """Code analysis results"""
    complexity: int
    lines_of_code: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    variables: List[str]
    errors: List[str]
    warnings: List[str]

@dataclass
class ExecutionResult:
    """Code execution results"""
    success: bool
    output: str
    error: str
    execution_time: float
    return_code: int

class UnifiedProgrammingModel:
    """
    Unified Programming Model
    Supports code analysis, execution, optimization, generation and other functions
    """
    
    def __init__(self, mode: str = "standard", config: Optional[Dict] = None):
        self.mode = mode
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Supported programming languages
        self.supported_languages = {
            "python": ".py",
            "javascript": ".js",
            "java": ".java",
            "cpp": ".cpp",
            "c": ".c"
        }
        
        # Execution environment
        self.execution_timeout = self.config.get("execution_timeout", 30)
        self.max_code_length = self.config.get("max_code_length", 10000)
        
        # Cache
        self.analysis_cache = {}
        self.cache_size = 100
        
    def analyze_code(self, code: str, language: str = "python") -> CodeAnalysis:
        """Analyze code"""
        if language not in self.supported_languages:
            return CodeAnalysis(
                complexity=0, lines_of_code=0, functions=[], classes=[], 
                imports=[], variables=[], errors=[f"Unsupported language: {language}"], warnings=[]
            )
        
        # Check cache
        cache_key = hash(code + language)
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        try:
            if language == "python":
                analysis = self._analyze_python_code(code)
            else:
                analysis = self._analyze_generic_code(code, language)
            
            # Cache results
            if len(self.analysis_cache) >= self.cache_size:
                self.analysis_cache.pop(next(iter(self.analysis_cache)))
            self.analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            return CodeAnalysis(
                complexity=0, lines_of_code=0, functions=[], classes=[], 
                imports=[], variables=[], errors=[str(e)], warnings=[]
            )
    
    def execute_code(self, code: str, language: str = "python", 
                    input_data: str = "") -> ExecutionResult:
        """Execute code"""
        if language not in self.supported_languages:
            return ExecutionResult(
                success=False, output="", error=f"Unsupported language: {language}",
                execution_time=0.0, return_code=-1
            )
        
        try:
            if language == "python":
                return self._execute_python_code(code, input_data)
            else:
                return self._execute_external_code(code, language, input_data)
                
        except Exception as e:
            return ExecutionResult(
                success=False, output="", error=str(e),
                execution_time=0.0, return_code=-1
            )
    
    def optimize_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Optimize code"""
        analysis = self.analyze_code(code, language)
        
        optimizations = []
        
        # Basic optimization suggestions
        if analysis.complexity > 10:
            optimizations.append({
                "type": "complexity",
                "message": "Function complexity is too high, recommend splitting",
                "severity": "medium"
            })
        
        if analysis.lines_of_code > 100:
            optimizations.append({
                "type": "length",
                "message": "Too many lines of code, recommend modularization",
                "severity": "low"
            })
        
        # Advanced optimization in enhanced mode
        if self.mode == "enhanced":
            optimizations.extend(self._enhanced_optimization(code, language))
        
        return {
            "original_complexity": analysis.complexity,
            "original_lines": analysis.lines_of_code,
            "optimizations": optimizations,
            "suggestions": self._generate_suggestions(code, language)
        }
    
    def generate_code_template(self, template_type: str, 
                              parameters: Dict[str, Any] = None) -> str:
        """Generate code template"""
        parameters = parameters or {}
        
        templates = {
            "function": self._generate_function_template,
            "class": self._generate_class_template,
            "web_api": self._generate_web_api_template,
            "data_processing": self._generate_data_processing_template
        }
        
        if template_type not in templates:
            return f"# Unsupported template type: {template_type}"
        
        return templates[template_type](parameters)
    
    def validate_syntax(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Validate code syntax"""
        try:
            if language == "python":
                compile(code, '<string>', 'exec')
                return {"valid": True, "errors": []}
            else:
                return self._validate_external_syntax(code, language)
                
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [{
                    "line": e.lineno,
                    "message": str(e),
                    "offset": e.offset
                }]
            }
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}
    
    def get_code_metrics(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Get code metrics"""
        analysis = self.analyze_code(code, language)
        
        return {
            "lines_of_code": analysis.lines_of_code,
            "cyclomatic_complexity": analysis.complexity,
            "function_count": len(analysis.functions),
            "class_count": len(analysis.classes),
            "import_count": len(analysis.imports),
            "variable_count": len(analysis.variables),
            "error_count": len(analysis.errors),
            "warning_count": len(analysis.warnings)
        }
    
    def _analyze_python_code(self, code: str) -> CodeAnalysis:
        """Analyze Python code"""
        lines = code.strip().split('\n')
        lines_of_code = len([line for line in lines if line.strip()])
        
        try:
            tree = ast.parse(code)
            
            functions = []
            classes = []
            imports = []
            variables = []
            errors = []
            warnings = []
            
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                    # Calculate cyclomatic complexity
                    complexity += sum(1 for n in ast.walk(node) 
                                    if isinstance(n, (ast.If, ast.While, ast.For, ast.ExceptHandler)))
                
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.dump(node))
                
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.append(target.id)
            
            return CodeAnalysis(
                complexity=complexity,
                lines_of_code=lines_of_code,
                functions=functions,
                classes=classes,
                imports=imports,
                variables=variables,
                errors=errors,
                warnings=warnings
            )
            
        except SyntaxError as e:
            return CodeAnalysis(
                complexity=0, lines_of_code=lines_of_code,
                functions=[], classes=[], imports=[], variables=[],
                errors=[str(e)], warnings=[]
            )
    
    def _analyze_generic_code(self, code: str, language: str) -> CodeAnalysis:
        """Analyze generic code"""
        lines = code.strip().split('\n')
        lines_of_code = len([line for line in lines if line.strip()])
        
        # Simple pattern matching
        function_pattern = r'def\s+(\w+)\s*\(|function\s+(\w+)\s*\(|(\w+)\s*\(\s*\)\s*\{'
        class_pattern = r'class\s+(\w+)|struct\s+(\w+)'
        import_pattern = r'import\s+|#include\s+|require\s+'
        
        functions = re.findall(function_pattern, code)
        classes = re.findall(class_pattern, code)
        imports = re.findall(import_pattern, code)
        
        return CodeAnalysis(
            complexity=lines_of_code // 10 + 1,
            lines_of_code=lines_of_code,
            functions=[f[0] or f[1] or f[2] for f in functions],
            classes=[c[0] or c[1] for c in classes],
            imports=imports,
            variables=[],
            errors=[],
            warnings=[]
        )
    
    def _execute_python_code(self, code: str, input_data: str = "") -> ExecutionResult:
        """Execute Python code"""
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        start_time = time.time()
        
        try:
            # Create execution environment
            local_vars = {}
            
            # Redirect input/output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, {"__builtins__": __builtins__}, local_vars)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                output=stdout_capture.getvalue(),
                error=stderr_capture.getvalue(),
                execution_time=execution_time,
                return_code=0
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
                return_code=1
            )
    
    def _execute_external_code(self, code: str, language: str, 
                             input_data: str = "") -> ExecutionResult:
        """Execute external code"""
        extension = self.supported_languages.get(language, ".txt")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=extension, 
                                       delete=False) as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name
        
        try:
            start_time = time.time()
            
            # Select executor based on language
            if language == "javascript":
                result = subprocess.run(
                    ["node", temp_path],
                    input=input_data.encode() if input_data else None,
                    capture_output=True, text=True, timeout=self.execution_timeout
                )
            elif language == "java":
                # Compile and execute Java
                subprocess.run(["javac", temp_path], check=True)
                class_name = os.path.basename(temp_path).replace('.java', '')
                result = subprocess.run(
                    ["java", class_name],
                    cwd=os.path.dirname(temp_path),
                    input=input_data.encode() if input_data else None,
                    capture_output=True, text=True, timeout=self.execution_timeout
                )
            else:
                return ExecutionResult(
                    success=False, output="", 
                    error=f"Additional configuration required to execute {language} code",
                    execution_time=0.0, return_code=-1
                )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                execution_time=execution_time,
                return_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False, output="", error="Execution timeout",
                execution_time=self.execution_timeout, return_code=-1
            )
        except Exception as e:
            return ExecutionResult(
                success=False, output="", error=str(e),
                execution_time=time.time() - start_time, return_code=1
            )
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _enhanced_optimization(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Advanced optimization in enhanced mode"""
        optimizations = []
        
        # Memory optimization suggestions
        if "for" in code and "range" in code:
            optimizations.append({
                "type": "memory",
                "message": "Consider using generator expressions to reduce memory usage",
                "severity": "low"
            })
        
        # Performance optimization suggestions
        if "list.append" in code:
            optimizations.append({
                "type": "performance",
                "message": "Consider using list comprehensions to improve performance",
                "severity": "low"
            })
        
        return optimizations
    
    def _generate_suggestions(self, code: str, language: str) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if language == "python":
            if "import *" in code:
                suggestions.append("Avoid using 'from module import *'")
            
            if len(code.split('\n')) > 50 and "def " not in code:
                suggestions.append("Consider splitting code into functions")
            
            if "print(" in code and __name__ != "__main__":
                suggestions.append("Consider using logging instead of print")
        
        return suggestions
    
    def _generate_function_template(self, params: Dict[str, Any]) -> str:
        """Generate function template"""
        func_name = params.get("name", "my_function")
        params_list = params.get("parameters", [])
        return_type = params.get("return_type", "None")
        
        params_str = ", ".join(params_list)
        
        return f'''def {func_name}({params_str}):
    """
    {params.get('description', 'Function description')}
    
    Parameters:
{chr(10).join([f'        {p}: Parameter description' for p in params_list])}
    
    Returns:
        {return_type}: Return value description
    """
    # TODO: Implement function logic
    return None'''
    
    def _generate_class_template(self, params: Dict[str, Any]) -> str:
        """Generate class template"""
        class_name = params.get("name", "MyClass")
        
        return f'''class {class_name}:
    """
    {params.get('description', 'Class description')}
    """
    
    def __init__(self{', ' + params.get('init_params', '') if params.get('init_params') else ''}):
        """
        Initialize
        """
{chr(10).join([f'        self.{p.split("=")[0].strip()} = {p.split("=")[0].strip()}' for p in params.get('attributes', [])])}
    
    def __str__(self):
        return f"{class_name}()"'''
    
    def _generate_web_api_template(self, params: Dict[str, Any]) -> str:
        """Generate Web API template"""
        return '''from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/endpoint', methods=['GET', 'POST'])
def api_endpoint():
    """
    API endpoint description
    """
    if request.method == 'GET':
        return jsonify({"message": "GET request successful"})
    
    data = request.get_json()
    # TODO: Handle POST request
    return jsonify({"message": "POST request successful", "data": data})

if __name__ == '__main__':
    app.run(debug=True)'''
    
    def _generate_data_processing_template(self, params: Dict[str, Any]) -> str:
        """Generate data processing template"""
        return '''import pandas as pd
import numpy as np

def process_data(data):
    """
    Data processing function
    """
    # Data cleaning
    data_cleaned = data.dropna()
    
    # Feature engineering
    # TODO: Add feature processing logic
    
    # Data analysis
    summary = data_cleaned.describe()
    
    return {
        'cleaned_data': data_cleaned,
        'summary': summary,
        'processed': True
    }

# Usage example
# df = pd.read_csv('data.csv')
# result = process_data(df)'''
    
    def _validate_external_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """Validate external language syntax"""
        try:
            extension = self.supported_languages.get(language, ".txt")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix=extension, 
                                           delete=False) as temp_file:
                temp_file.write(code)
                temp_path = temp_file.name
            
            # Use external tools for syntax validation
            if language == "javascript":
                result = subprocess.run(
                    ["node", "--check", temp_path],
                    capture_output=True, text=True
                )
            elif language == "java":
                result = subprocess.run(
                    ["javac", "-Xlint", temp_path],
                    capture_output=True, text=True
                )
            else:
                return {"valid": True, "errors": ["Syntax validation not available"]}
            
            if result.returncode == 0:
                return {"valid": True, "errors": []}
            else:
                return {"valid": False, "errors": [result.stderr]}
                
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
