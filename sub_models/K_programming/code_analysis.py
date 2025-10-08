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

# Programming Model - Code Analysis and Improvement

import ast
import astunparse
import os
import re
import subprocess
from .knowledge_integration import KnowledgeIntegration

class CodeAnalyzer:
    def __init__(self):
        """Initialize code analyzer"""
        self.knowledge = KnowledgeIntegration()
        self.code_patterns = self._load_code_patterns()
        
    def _load_code_patterns(self):
        """Load code pattern knowledge"""
        patterns = {
            "design_patterns": self.knowledge.get_design_patterns(),
            "best_practices": self.knowledge.get_best_practices(),
            "common_errors": self.knowledge.get_common_errors(),
            "performance_optimizations": self.knowledge.get_performance_optimizations()
        }
        return patterns
        
    def analyze_code(self, file_path):
        """Analyze code and provide improvement suggestions"""
        if not os.path.exists(file_path):
            return {"error": f"File does not exist: {file_path}"}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            
        # Perform code analysis
        analysis_result = {
            "file": file_path,
            "suggestions": [],
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        # 1. Static code analysis
        self._static_analysis(code, analysis_result)
        
        # 2. Code complexity analysis
        self._complexity_analysis(code, analysis_result)
        
        # 3. Design pattern application check
        self._design_pattern_check(code, analysis_result)
        
        # 4. Best practices check
        self._best_practice_check(code, analysis_result)
        
        return analysis_result
        
    def _static_analysis(self, code, result):
        """Perform static code analysis"""
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Check for unused imports
            self._check_unused_imports(tree, result)
            
            # Check for unused variables
            self._check_unused_variables(tree, result)
            
            # Check for long functions
            self._check_long_functions(tree, result)
            
        except SyntaxError as e:
            result["errors"].append(f"Syntax error: {e.msg} (line {e.lineno})")
            
    def _complexity_analysis(self, code, result):
        """Analyze code complexity"""
        # Calculate cyclomatic complexity
        complexity = self._calculate_cyclomatic_complexity(code)
        result["metrics"]["cyclomatic_complexity"] = complexity
        
        if complexity > 10:
            result["warnings"].append(f"High cyclomatic complexity: {complexity} (suggest refactoring function)")
            
    def _design_pattern_check(self, code, result):
        """Check design pattern application"""
        for pattern_name, pattern_info in self.code_patterns["design_patterns"].items():
            pattern_regex = pattern_info.get("detection_regex")
            if pattern_regex and re.search(pattern_regex, code):
                result["suggestions"].append({
                    "type": "Design pattern application",
                    "pattern": pattern_name,
                    "description": pattern_info["description"],
                    "recommendation": f"Consider applying {pattern_name} pattern to improve code maintainability"
                })
                
    def _best_practice_check(self, code, result):
        """Check best practices"""
        for practice_name, practice_info in self.code_patterns["best_practices"].items():
            violation_regex = practice_info.get("violation_regex")
            if violation_regex and re.search(violation_regex, code):
                result["suggestions"].append({
                    "type": "Best practice",
                    "practice": practice_name,
                    "description": practice_info["description"],
                    "recommendation": practice_info["recommendation"]
                })
                
    def refactor_code(self, file_path, suggestions):
        """Refactor code based on suggestions"""
        if not os.path.exists(file_path):
            return {"error": f"File does not exist: {file_path}"}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            
        # Apply refactoring
        for suggestion in suggestions:
            if suggestion["type"] == "Design pattern application":
                code = self._apply_design_pattern(code, suggestion["pattern"])
            elif suggestion["type"] == "Best practice":
                code = self._apply_best_practice(code, suggestion["practice"])
                
        # Save refactored code
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
            
        return {"status": "success", "message": "Code refactoring completed"}
        
    def _apply_design_pattern(self, code, pattern_name):
        """Apply design pattern"""
        pattern_info = self.code_patterns["design_patterns"].get(pattern_name)
        if not pattern_info:
            return code
            
        # In actual implementation, there would be more complex pattern application logic
        # Here simplified to adding pattern comments
        return f"# Refactored with {pattern_name} pattern\n{code}"
        
    def _apply_best_practice(self, code, practice_name):
        """Apply best practice"""
        practice_info = self.code_patterns["best_practices"].get(practice_name)
        if not practice_info:
            return code
            
        # In actual implementation, there would be more complex refactoring logic
        # Here simplified to adding best practice comments
        return f"# Applied {practice_name} best practice\n{code}"
        
    def auto_improve_system(self):
        """Automatically improve the entire system"""
        # 1. Identify modules for improvement
        modules_to_improve = self._identify_modules_for_improvement()
        
        # 2. Analyze each module
        improvement_plans = []
        for module in modules_to_improve:
            analysis = self.analyze_code(module)
            improvement_plans.append({
                "module": module,
                "analysis": analysis,
                "actions": self._generate_improvement_actions(analysis)
            })
            
        # 3. Execute improvements
        for plan in improvement_plans:
            self._execute_improvement_actions(plan["module"], plan["actions"])
            
        return {
            "status": "success",
            "message": "System auto-improvement completed",
            "improved_modules": [plan["module"] for plan in improvement_plans]
        }
        
    def _identify_modules_for_improvement(self):
        """Identify modules for improvement"""
        # In actual system, there would be more complex logic to identify modules for improvement
        # Here simplified to return a fixed list
        return [
            "manager_model/core_system.py",
            "training_manager/train_scheduler.py",
            "sub_models/I_knowledge/knowledge_model.py"
        ]
        
    def _generate_improvement_actions(self, analysis):
        """Generate improvement actions"""
        actions = []
        for suggestion in analysis["suggestions"]:
            actions.append({
                "type": suggestion["type"],
                "target": suggestion.get("pattern") or suggestion.get("practice"),
                "description": suggestion["recommendation"]
            })
        return actions
        
    def _execute_improvement_actions(self, file_path, actions):
        """Execute improvement actions"""
        for action in actions:
            if action["type"] == "Design pattern application":
                self.refactor_code(file_path, [{
                    "type": "Design pattern application",
                    "pattern": action["target"]
                }])
            elif action["type"] == "Best practice":
                self.refactor_code(file_path, [{
                    "type": "Best practice",
                    "practice": action["target"]
                }])
                
    def _calculate_cyclomatic_complexity(self, code):
        """Calculate cyclomatic complexity"""
        # Simplified implementation - should use more precise methods in actual system
        # Based on the number of control flow statements
        control_flow_keywords = ["if", "elif", "else", "for", "while", "and", "or", "except"]
        complexity = 1
        for keyword in control_flow_keywords:
            complexity += code.count(keyword)
        return complexity
        
    def _check_unused_imports(self, tree, result):
        """Check for unused imports"""
        # To be implemented in actual system
        pass
        
    def _check_unused_variables(self, tree, result):
        """Check for unused variables"""
        # To be implemented in actual system
        pass
        
    def _check_long_functions(self, tree, result):
        """Check for long functions"""
        # To be implemented in actual system
        pass

class KnowledgeIntegration:
    def __init__(self):
        """Initialize knowledge integration"""
        self.design_patterns = self._load_design_patterns()
        self.best_practices = self._load_best_practices()
        
    def _load_design_patterns(self):
        """Load design pattern knowledge"""
        return {
            "Singleton": {
                "description": "Ensure a class has only one instance and provide a global access point",
                "detection_regex": r"class .+\(.*\):.*def __new__\(.*\):",
                "recommendation": "Use singleton pattern to manage global state"
            },
            "Observer": {
                "description": "Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified",
                "detection_regex": r"\.attach\(|\.detach\(|\.notify\(",
                "recommendation": "Use observer pattern to implement event notification system"
            },
            "Factory": {
                "description": "Define an interface for creating an object, but let subclasses decide which class to instantiate",
                "detection_regex": r"def create_.+\(",
                "recommendation": "Use factory pattern to encapsulate object creation logic"
            }
        }
        
    def _load_best_practices(self):
        """Load best practices knowledge"""
        return {
            "DRY": {
                "description": "Don't Repeat Yourself principle",
                "violation_regex": r"重复的代码块",
                "recommendation": "Extract duplicate code into functions or classes"
            },
            "MeaningfulNames": {
                "description": "Use meaningful variable names",
                "violation_regex": r"def [a-z]\(| = [a-z] | [a-z] = ",
                "recommendation": "Use descriptive variable names to improve code readability"
            },
            "ShortFunctions": {
                "description": "Keep functions short",
                "violation_regex": r"def .+\(.*\):(\n\s{4}.{50,}){10}",
                "recommendation": "Split long functions into multiple smaller functions"
            }
        }
        
    def get_design_patterns(self):
        """Get design patterns knowledge"""
        return self.design_patterns
        
    def get_best_practices(self):
        """Get best practices knowledge"""
        return self.best_practices

if __name__ == '__main__':
    # Test code analyzer
    analyzer = CodeAnalyzer()
    
    # Analyze example file
    analysis = analyzer.analyze_code("example.py")
    print("Code analysis result:", analysis)
    
    # Auto-improve system
    improvement_result = analyzer.auto_improve_system()
    print("System improvement result:", improvement_result)
