# Copyright 2025 AGI System Team
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


class ProgrammingModel:
    """Programming Model - Capable of autonomous programming and system improvement
    """
    
    # AGI System Core Component

    def __init__(self, knowledge_base=None, language='en'):
        # New feature implementation
        self.language = language
        self.knowledge_base = knowledge_base  # Knowledge base expert model reference
        self.codebase_map = {}
        self.project_dependencies = {}
        self.code_history = []
        self.last_self_improvement = 0
        self.self_improvement_interval = 3600  # Self-improvement every hour
        self.model_environment_map = {}
        self.supported_languages = [
            'python', 'javascript', 'java', 'c++', 'go']
        self.test_frameworks = {
            'python': 'pytest',
            'javascript': 'jest',
            'java': 'junit',
            'c++': 'gtest',
            'go': 'testing'
        }

        # Initialize codebase map
        self._build_codebase_map()

        # Analyze project dependencies
        self._analyze_project_dependencies()

        # Start self-improvement monitoring thread
        self._start_self_improvement_monitor()

    def _build_codebase_map(self):
        """Build codebase map, recording the location and structure of all model files"""
        try:
            # Project root directory
            project_root = Path(__file__).resolve().parent.parent.parent

            # Build sub-models codebase
            self._build_sub_models_codebase(project_root)

            # Build manager model codebase
            self._build_manager_model_codebase(project_root)

            # Add programming model itself to model environment mapping
            self._add_self_to_model_environment()

            # Analyze dependencies of all models
            self._analyze_all_model_dependencies()

            logger.info("Codebase map construction completed")
            return True
        except Exception as e:
            logger.error(f"Error building codebase map: {str(e)}")
            return False

    def _add_self_to_model_environment(self):
        """Add programming model itself to model environment mapping"""
        model_id = 'K'  # Programming model ID
        model_path = Path(__file__).resolve().parent

        self.model_environment_map[model_id] = {
            'path': str(model_path),
            'main_file': str(model_path / 'app.py'),
            'dependencies': [],
            'config_files': []
        }
        logger.info(f"Added programming model itself to model environment: {model_id}")

    def _build_sub_models_codebase(self, project_root):
        """Build sub-models codebase"""
        models_dir = project_root / 'sub_models'
        if not models_dir.exists():
            return

        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_id = self._get_model_id_from_dir(model_dir)
            self._initialize_model_environment(model_id, model_dir)
            self._find_model_files(model_id, model_dir)

    def _get_model_id_from_dir(self, model_dir):
        """Get model ID from directory name"""
        if '_' in model_dir.name:
            return model_dir.name.split('_')[1].upper()
        return model_dir.name.upper()

    def _initialize_model_environment(self, model_id, model_dir):
        """Initialize model environment"""
        self.model_environment_map[model_id] = {
            'path': str(model_dir),
            'main_file': None,
            'dependencies': [],
            'config_files': []
        }

    def _find_model_files(self, model_id, model_dir):
        """Find model-related files"""
        for file in model_dir.rglob('*'):
            if not file.is_file():
                continue

            if file.name == 'app.py' or file.name == '__main__.py':
                self.model_environment_map[model_id]['main_file'] = str(file)
            elif file.suffix in ['.json', '.yml', '.yaml']:
                self.model_environment_map[model_id]['config_files'].append(
                    str(file))

    def _build_manager_model_codebase(self, project_root):
        """Build manager model codebase"""
        manager_dir = project_root / 'manager_model'
        if not manager_dir.exists():
            return

        self._initialize_model_environment('A', manager_dir)
        self._find_manager_model_files(manager_dir)

    def _find_manager_model_files(self, manager_dir):
        """Find manager model files"""
        for file in manager_dir.rglob('*'):
            if not file.is_file():
                continue

            if file.name == 'core_system.py':
                self.model_environment_map['A']['main_file'] = str(file)
            elif file.suffix in ['.json', '.yml', '.yaml']:
                self.model_environment_map['A']['config_files'].append(
                    str(file))

    def _analyze_project_dependencies(self):
        """Analyze project dependencies"""
        try:
            # Read requirements.txt from project root
            project_root = Path(__file__).resolve().parent.parent.parent
            req_file = project_root / 'requirements.txt'

            if req_file.exists():
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                        # Simple dependency parsing
                            dep = line.split('==')[0].strip()
                            self.project_dependencies[dep] = line

            logger.info("Project dependency analysis completed")
            return True
        except Exception as e:
            logger.error(f"Error analyzing project dependencies: {str(e)}")
            return False

    def _analyze_all_model_dependencies(self):
        """Analyze dependencies of all models"""
        try:
            for model_id, model_env in self.model_environment_map.items():
                self._analyze_model_dependencies(model_id, model_env)
            logger.info("All model dependency analysis completed")
            return True
        except Exception as e:
            logger.error(f"Error analyzing model dependencies: {str(e)}")
            return False

    def get_all_model_dependencies(self) -> Dict[str, Dict[str, str]]:
        """Get dependencies for all models (with version information)
        """
        dependencies = {}
        for model_id, model_env in self.model_environment_map.items():
            dependencies[model_id] = self.get_model_dependencies(model_id)
        return dependencies

    def get_model_dependencies(self, model_id: str) -> Dict[str, str]:
        """Get dependencies for a specific model (with version information)
        """
        if model_id not in self.model_environment_map:
            return {}

        model_env = self.model_environment_map[model_id]
        model_path = Path(model_env['path'])
        dependency_files = self._find_dependency_files(model_path)

        dependencies = {}
        for dep_file in dependency_files:
            deps = self._collect_dependencies_from_file(dep_file)
            dependencies.update(deps)

        # Add inter-model dependencies
        dependencies.update(self._get_model_interdependencies(model_id))

        return dependencies

    def _get_model_interdependencies(self, model_id: str) -> Dict[str, str]:
        """Get inter-model dependencies
        """
        inter_deps = {}
        if model_id == 'A':  # Manager model depends on all sub-models
            for submodel_id in self.model_environment_map.keys():
                if submodel_id != 'A':
                    inter_deps[f"model_{submodel_id}"] = "required"
        # Other inter-model dependencies can be added here
        return inter_deps

    def _analyze_model_dependencies(self, model_id, model_env):
        """Analyze single model dependencies (enhanced version)"""
        # Now this method calls the enhanced get_model_dependencies
        self.get_model_dependencies(model_id)

    def _find_dependency_files(self, model_path):
        """Find dependency files in model directory (enhanced version)"""
        dependency_files = []

        # Supported file types (extended list)
        supported_files = [
            'requirements.txt',
            'package.json',
            'pom.xml',
            'build.gradle',
            'build.sbt',
            'Gemfile',
            'Cargo.toml',
            'go.mod',
            'dependencies.yaml',
            'dependencies.yml'
        ]

        for file_name in supported_files:
            file_path = model_path / file_name
            if file_path.exists():
                dependency_files.append(file_path)

        return dependency_files

    def _collect_dependencies_from_file(self, file_path) -> Dict[str, str]:
        """Collect dependencies from a single dependency file (enhanced)
        """
        # Dispatch to specialized parser based on file type
        file_name = file_path.name
        if file_name == 'requirements.txt':
            return self._parse_requirements_txt(file_path)
        elif file_name == 'package.json':
            return self._parse_package_json(file_path)
        else:
            return self._parse_other_dependency_files(file_path)
            
    def _parse_requirements_line(self, line: str) -> Tuple[str, str]:
        """Parse a single dependency line from requirements.txt
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None
            
        if '==' in line:
            parts = line.split('==')
            dep_name = parts[0].strip()
            dep_version = parts[1].split(';')[0].strip() if len(parts) > 1 else 'latest'
        elif '>=' in line:
            parts = line.split('>=')
            dep_name = parts[0].strip()
            dep_version = f">={parts[1].split(';')[0].strip()}" if len(parts) > 1 else 'latest'
        else:
            dep_name = line.split(';')[0].strip()
            dep_version = 'latest'
            
        return dep_name, dep_version

    def _parse_requirements_txt(self, file_path) -> Dict[str, str]:
        """Parse requirements.txt file
        """
        dependencies = {}
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    dep_name, dep_version = self._parse_requirements_line(line)
                    if dep_name:
                        dependencies[dep_name] = dep_version
                        self.project_dependencies[dep_name] = line
            return dependencies
        except Exception as e:
            logger.error(f"Error parsing requirements.txt: {str(e)}")
            return {}
            
    def _parse_package_json(self, file_path) -> Dict[str, str]:
        """Parse package.json file
        """
        try:
            with open(file_path, 'r') as f:
                pkg_data = json.load(f)
                dependencies = {}
                for dep_type in ['dependencies', 'devDependencies', 'peerDependencies']:
                    if dep_type in pkg_data:
                        for dep_name, version in pkg_data[dep_type].items():
                            dependencies[dep_name] = version
                return dependencies
        except Exception as e:
            logger.error(f"Error parsing package.json: {str(e)}")
            return {}
            
    def _parse_other_dependency_files(self, file_path) -> Dict[str, str]:
        """Parse other types of dependency files
        """
        logger.warning(f"Unsupported dependency file type: {file_path.name}")
        return {}

    def _improve_file(self, model_id, file_path, requirements, file_type):
        """Improve a single file (refactored version)
        """
        try:
            # Analyze file content
            analysis = self._analyze_file(file_path)
            if 'error' in analysis:
                return [{'error': analysis['error']}]

            # Generate improvement plan
            improvements = self._generate_file_improvements(
                file_path,
                analysis,
                requirements
            )

            # Apply improvement plan
            applied_improvements = self._apply_improvements_to_file(
                file_path, improvements)

            # Record improvement history
            self._record_improvement_history(
                model_id,
                file_path,
                file_type,
                requirements,
                applied_improvements
            )

            return applied_improvements
        except Exception as e:
            logger.error(f"File improvement error: {str(e)}")
            return [{'error': str(e)}]

    def _apply_improvements_to_file(self, file_path, improvements):
        """Apply all improvements to the file (refactored version)
        """
        applied_improvements = []
        for imp in improvements:
            try:
                mod_result = self.modify_code(
                    file_path,
                    [imp['modification']],
                    reason=imp['reason']
                )
                if 'error' in mod_result:
                    imp['error'] = mod_result['error']
                else:
                    imp['test_result'] = mod_result.get('test_result', {})
                applied_improvements.append(imp)
            except Exception as e:
                logger.error(f"Failed to apply improvement: {str(e)}")
                imp['error'] = str(e)
                applied_improvements.append(imp)
        return applied_improvements

    def _record_improvement_history(
            self,
            model_id,
            file_path,
            file_type,
            requirements,
            improvements):
        """Record improvement history"""
        self.code_history.append({
            'timestamp': time.time(),
            'type': 'improve',
            'model_id': model_id,
            'file_path': file_path,
            'file_type': file_type,
            'requirements': requirements,
            'improvements': improvements
        })

    def _generate_python_code(
            self,
            requirements: str,
            framework: Optional[str],
            knowledge_assist: Dict) -> str:
        """Generate Python code (refactored version)
        """
        try:
            # Get base code template
            code_template = self._get_python_base_template()
            
            # Process knowledge base suggestions
            knowledge_suggestions = self._process_knowledge_suggestions(knowledge_assist)
            
            # Get framework-specific code
            framework_code = self._get_framework_specific_code(framework)
            
            # Assemble final code
            return self._assemble_python_code(
                code_template, 
                requirements, 
                knowledge_suggestions, 
                framework_code
            )
        except Exception as e:
            error_msg = f"Error generating Python code: {str(e)}"
            logger.error(error_msg)
            return f"# {error_msg}\n# Requirements: {requirements}"

    def _get_python_base_template(self) -> str:
        """Get Python base code template"""
        return """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{requirements}

Knowledge base optimization suggestions:
{knowledge_suggestions}
\"\"\"

import os
import sys
import logging
from typing import Dict, List, Optional, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeneratedCode:
    \"\"\"Generated code class - Core component of AGI system\"\"\"

    def __init__(self, config: Optional[Dict] = None):
        \"\"\"Initialize generated code

        Parameters:
        - config: Configuration dictionary
        \"\"\"
        self.config = config or {}
        self.data: Dict[str, Any] = {}
        self._setup()

    def _setup(self) -> None:
        \"\"\"Initialization setup\"\"\"
        # Initialize based on configuration
        if 'debug' in self.config and self.config['debug']:
            logger.setLevel(logging.DEBUG)

    def execute(self) -> Any:
        \"\"\"Execute main logic - This is the core functionality of the code

        Returns:
        - Execution result
        \"\"\"
        try:
            # Main execution logic
            logger.info("Starting generated code execution")
            result = self._process_requirements()
            logger.info(f"Execution completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            self._handle_error(e)
            raise

    def _process_requirements(self) -> Any:
        \"\"\"Process requirements logic - Implement based on specific requirements

        Returns:
        - Processing result
        \"\"\"
        # This is the specific logic implemented based on requirements
        return "Execution successful"

    def _handle_error(self, error: Exception) -> None:
        \"\"\"Error handling - Unified error handling mechanism

        Parameters:
        - error: Exception object
        \"\"\"
        # Log detailed error information
        logger.error(f"Error details: {str(error)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")

    def cleanup(self) -> None:
        \"\"\"Cleanup resources - Release all occupied resources\"\"\"
        self.data.clear()
        logger.info("Resources cleaned up")

# Main execution entry - AGI system entry point
if __name__ == "__main__":
    import traceback
    generator = GeneratedCode()
    try:
        result = generator.execute()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
    finally:
        generator.cleanup()
"""

    def _process_knowledge_suggestions(self, knowledge_assist: Dict) -> str:
        """Process knowledge base optimization suggestions"""
        if knowledge_assist and 'suggestions' in knowledge_assist:
            suggestions = knowledge_assist['suggestions']
            return "\n".join([f"# - {suggestion}" for suggestion in suggestions])
        return "# No knowledge base suggestions available"

    def _get_framework_specific_code(self, framework: Optional[str]) -> str:
        """Get framework-specific code"""
        if not framework:
            return ""
            
        framework_code = f"\n# Using {framework} framework"
        
        if framework.lower() == 'flask':
            framework_code += """
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return "AGI System Running"
"""
        elif framework.lower() == 'django':
            framework_code += """
# Django specific imports would be added here
"""
        
        return framework_code

    def _assemble_python_code(
            self,
            template: str,
            requirements: str,
            knowledge_suggestions: str,
            framework_code: str) -> str:
        """Assemble Python code"""
        # Insert framework code
        if framework_code:
            template = template.replace(
                "# Main execution logic", f"# Main execution logic{framework_code}")
        
        # Format final code
        return template.format(
            requirements=requirements,
            knowledge_suggestions=knowledge_suggestions
        )

    def _generate_javascript_code(
            self,
            requirements: str,
            framework: Optional[str],
            knowledge_assist: Dict) -> str:
        """Generate JavaScript code"""
        try:
            code_template = """/**
 * {requirements}
 */

// Browser-compatible implementation
class GeneratedCode {{
    constructor() {{
        this.config = {{}};
        this.data = {{}};
    }}

    execute() {{
        try {{
            console.log('Starting generated code execution');
            const result = this._processRequirements();
            console.log(`Execution completed: ${{result}}`);
            return result;
        }} catch (error) {{
            console.error(`Execution error: ${{error.message}}`);
            throw error;
        }}
    }}

    _processRequirements() {{
        // This is the specific logic implemented based on requirements
        return 'Execution successful';
    }}

    cleanup() {{
        this.data = {{}};
    }}
}}

// Main execution entry in browser environment
if (typeof window !== 'undefined') {{
    // Browser environment
    window.GeneratedCode = GeneratedCode;
}} else if (typeof module !== 'undefined' && module.exports) {{
    // Node.js environment
    module.exports = GeneratedCode;
}}

// Usage example
const generator = new GeneratedCode();
try {{
    const result = generator.execute();
    console.log(`Result: ${{result}}`);
}} finally {{
    generator.cleanup();
}}
"""
            return code_template.format(requirements=requirements)

        except Exception as e:
            logger.error(f"Error generating JavaScript code: {str(e)}")
            return f"// Code generation error: {str(e)}\n// Requirements: {requirements}"

    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze code file (refactored version)"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Perform basic analysis
            analysis = self._perform_basic_analysis(file_path, content)

            # Perform AST analysis (if possible)
            if not analysis.get('error'):
                self._perform_ast_analysis(content, analysis)

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing file: {str(e)}")
            return {'error': str(e)}

    def _perform_basic_analysis(
            self, file_path: str, content: str) -> Dict[str, Any]:
        """Perform basic file analysis"""
        return {
            'file_path': file_path,
            'line_count': len(content.split('\n')),
            'file_size': len(content),
            'complex_functions': [],
            'imports': [],
            'classes': [],
            'functions': []
        }

    def _perform_ast_analysis(self, content: str, analysis: Dict[str, Any]):
        """Perform code analysis using AST"""
        try:
            tree = ast.parse(content)
            self._analyze_imports(tree, analysis)
            self._analyze_functions_and_classes(tree, analysis)

        # Sort by complexity
            analysis['complex_functions'].sort(
                key=lambda x: x[1], reverse=True)
        except SyntaxError:
            logger.warning(f"File {analysis['file_path']} has syntax errors, cannot perform AST analysis")

    def _analyze_imports(self, tree: ast.AST, analysis: Dict[str, Any]):
        """Analyze import statements"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    analysis['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    analysis['imports'].append(f"{module}.{alias.name}")

    def _analyze_functions_and_classes(
            self, tree: ast.AST, analysis: Dict[str, Any]):
        """Analyze function and class definitions"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = len(list(ast.walk(node)))
                analysis['functions'].append({
                    'name': node.name,
                    'complexity': complexity,
                    'line_no': node.lineno
                })
                analysis['complex_functions'].append((node.name, complexity))

            elif isinstance(node, ast.ClassDef):
                analysis['classes'].append({
                    'name': node.name,
                    'line_no': node.lineno
                })

    def _generate_file_improvements(
            self,
            file_path: str,
            analysis: Dict,
            requirements: List[str]) -> List[Dict]:
        """Generate file improvement plan (refactored version)"""
        improvements = []

        # Improvements based on complex functions
        improvements += self._generate_complexity_improvements(analysis)

        # Improvements based on requirements
        improvements += self._generate_requirement_improvements(requirements)

        return improvements

    def _generate_complexity_improvements(self, analysis: Dict) -> List[Dict]:
        """Generate improvements based on function complexity"""
        improvements = []
        if not analysis.get('complex_functions'):
            return improvements

        # Only process the top 3 most complex functions
        for func_name, complexity in analysis['complex_functions'][:3]:
            try:
                if complexity > 50:  # Complexity threshold
                    improvements.append({
                        'reason': f'Refactor complex function {func_name} (complexity: {complexity})',
                        'modification': {
                            'find': f'def {func_name}(',
                            'replace': f'# TODO: Refactor {func_name} function to reduce complexity\ndef {func_name}('
                        }
                    })
            except Exception as e:
                logger.error(f"Error generating improvement for complex function {func_name}: {str(e)}")
        return improvements

    def _generate_requirement_improvements(
            self, requirements: List[str]) -> List[Dict]:
        """Generate improvements based on requirements (refactored version)"""
        # Use strategy pattern to handle different types of improvements
        improvement_strategies = {
            'performance': {
                'keywords': ['performance', 'optimization'],
                'handler': self._create_performance_improvement
            },
            'bug_fix': {
                'keywords': ['fix', 'bug'],
                'handler': self._create_bug_fix_improvement
            },
            'enhancement': {
                'keywords': ['enhance', 'feature'],
                'handler': self._create_enhancement_improvement
            }
        }
        
        improvements = []
        added_types = set()
        
        # Process each requirement
        for req in requirements:
            req_lower = req.lower()
            for imp_type, strategy in improvement_strategies.items():
                if imp_type in added_types:
                    continue
                    
                # Check if requirement contains keywords
                if any(keyword in req_lower for keyword in strategy['keywords']):
                    try:
                        improvements.append(strategy['handler']())
                        added_types.add(imp_type)
                        break  # Break inner loop after match found
                    except Exception as e:
                        logger.error(f"Error processing requirement '{req}': {str(e)}")
        
        return improvements

    def _create_performance_improvement(self) -> Dict:
        """Create performance optimization improvement item"""
        return {
            'reason': 'Performance optimization',
            'modification': {
                'insert_after': 'import ',
                'code': '# Performance optimization: Add caching mechanism or algorithm optimization\n'
            }
        }

    def _create_bug_fix_improvement(self) -> Dict:
        """Create bug fix improvement item"""
        return {
            'reason': 'Fix known issues',
            'modification': {
                'insert_before': 'class ',
                'code': '# Fix: Check boundary conditions and error handling\n'
            }
        }

    def _create_enhancement_improvement(self) -> Dict:
        """Create feature enhancement improvement item"""
        return {
            'reason': 'Feature enhancement',
            'modification': {
                'insert_after': 'def __init__',
                'code': '        # New feature implementation\n'
            }
        }

    def run_tests(self, target: str) -> Dict[str, Any]:
        """Run tests (supports multiple languages and model types)
        """
        try:
            # Select test framework based on target type
            if target.endswith('.py'):
                # Python model testing
                return self._run_python_tests(target)
            elif target.endswith('.js'):
                # JavaScript model testing
                return self._run_javascript_tests(target)
            elif target.endswith('.java'):
                # Java model testing
                return self._run_java_tests(target)
            elif target.endswith('.cpp') or target.endswith('.h'):
                # C++ model testing
                return self._run_cpp_tests(target)
            elif target.endswith('.go'):
                # Go model testing
                return self._run_go_tests(target)
            elif target.endswith('.json') or target.endswith('.yaml') or target.endswith('.yml'):
                # Configuration file testing
                return self._run_config_tests(target)
            else:
                # Model ID testing
                return self._run_model_tests(target)

        except subprocess.TimeoutExpired:
            return {'error': 'Test timeout'}
        except Exception as e:
            return {'error': f'Test execution error: {str(e)}'}

    def _run_python_tests(self, target: str) -> Dict[str, Any]:
        """Run Python model tests"""
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', target, '-v'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }

    def _run_javascript_tests(self, target: str) -> Dict[str, Any]:
        """Run JavaScript model tests"""
        result = subprocess.run(
            ['npx', 'jest', target, '--verbose'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }

    def _run_java_tests(self, target: str) -> Dict[str, Any]:
        """Run Java model tests"""
        # Compile and run JUnit tests
        class_dir = Path(target).parent
        result = subprocess.run(
            ['javac', '-d', str(class_dir), target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60
        )
        if result.returncode != 0:
            return {
                'success': False,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }

        test_class = Path(target).stem
        test_result = subprocess.run(
            ['java', '-cp', str(class_dir), 'org.junit.runner.JUnitCore', test_class],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60
        )
        return {
            'success': 'OK' in test_result.stdout,
            'stdout': test_result.stdout,
            'stderr': test_result.stderr,
            'returncode': test_result.returncode
        }

    def _run_cpp_tests(self, target: str) -> Dict[str, Any]:
        """Run C++ model tests"""
        # Compile test file
        test_binary = Path(target).with_suffix('.test')
        compile_result = subprocess.run(
            ['g++', target, '-o', str(test_binary), '-lgtest', '-lgtest_main', '-pthread'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=120
        )
        if compile_result.returncode != 0:
            return {
                'success': False,
                'stdout': compile_result.stdout,
                'stderr': compile_result.stderr,
                'returncode': compile_result.returncode
            }

        # Run test
        test_result = subprocess.run(
            [str(test_binary)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=30
        )
        return {
            'success': test_result.returncode == 0,
            'stdout': test_result.stdout,
            'stderr': test_result.stderr,
            'returncode': test_result.returncode
        }

    def _run_go_tests(self, target: str) -> Dict[str, Any]:
        """Run Go model tests"""
        result = subprocess.run(
            ['go', 'test', target, '-v'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }

    def _run_config_tests(self, target: str) -> Dict[str, Any]:
        """Run configuration file tests (validate syntax)"""
        try:
            with open(target, 'r') as f:
                if target.endswith('.json'):
                    json.load(f)
                elif target.endswith('.yaml') or target.endswith('.yml'):
                    import yaml
                    yaml.safe_load(f)
            return {'success': True, 'message': f'Configuration file {target} syntax validation passed'}
        except Exception as e:
            return {'success': False, 'error': f'Configuration file {target} syntax error: {str(e)}'}

    def _run_model_tests(self, model_id: str) -> Dict[str, Any]:
        """Run entire model tests"""
        if model_id not in self.model_environment_map:
            return {'error': f'Model does not exist: {model_id}'}

        model_env = self.model_environment_map[model_id]
        test_results = {}

        # Test main file
        if model_env['main_file']:
            test_results['main_file'] = self.run_tests(model_env['main_file'])

        # Test configuration files
        for config_file in model_env['config_files']:
            test_results[Path(config_file).name] = self.run_tests(config_file)

        # Check if all tests passed
        all_success = all(result.get('success', False)
                          for result in test_results.values())

        return {
            'success': all_success,
            'test_results': test_results,
            'model_id': model_id
        }

    def generate_code(self, requirements, language='python', framework=None):
        """Generate code based on requirements"""
        try:
            # Check if language is supported
            if language not in self.supported_languages:
                return {'error': f'Unsupported programming language: {language}'}

            # Get knowledge base assistance
            knowledge_assist = {}
            if self.knowledge_base:
                knowledge_assist = self.knowledge_base.assist_model(
        'K', f'Generate {language} code: {requirements}')

            # Generate code
            if language == 'python':
                code = self._generate_python_code(
                    requirements, framework, knowledge_assist)
            elif language == 'javascript':
                code = self._generate_javascript_code(
                    requirements, framework, knowledge_assist)
            else:
                code = f"# Code generation placeholder\n# Requirements: {requirements}"

            # Record code history
            self.code_history.append({
                'timestamp': time.time(),
                'type': 'generate',
                'requirements': requirements,
                'language': language,
                'code': code
            })

            return {
                'code': code,
                'language': language,
                'knowledge_assist': knowledge_assist}
        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            return {'error': str(e)}

    def modify_code(self, file_path, modifications, reason=None):
        """Modify existing code file (refactored version)"""
        # Validate file existence
        if not os.path.exists(file_path):
            return {'error': f'File does not exist: {file_path}'}

        try:
            # Execute code modification process
            return self._execute_code_modification(
                file_path, modifications, reason)
        except Exception as e:
            logger.error(f"Code modification error: {str(e)}")
            return {'error': str(e)}

    def _execute_code_modification(self, file_path, modifications, reason):
        """Execute code modification process"""
        # Read original code
        original_code = self._read_file_content(file_path)

        # Apply modifications
        modified_code = self._apply_modifications(original_code, modifications)

        # Write modified code
        self._write_file_content(file_path, modified_code)

        # Record modification history
        self._record_modification_history(
            file_path, original_code, modified_code, reason)

        # Run tests to verify modifications
        test_result = self.run_tests(file_path)

        return {
            'success': True,
            'file_path': file_path,
            'modifications': modifications,
            'test_result': test_result
        }

    def _read_file_content(self, file_path):
        """Read file content"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _apply_modifications(self, original_code, modifications):
        """Apply all modifications to code"""
        modified_code = original_code
        for mod in modifications:
            modified_code = self._apply_single_modification(modified_code, mod)
        return modified_code

    def _apply_single_modification(self, code, modification):
        """Apply single modification to code"""
        if 'find' in modification and 'replace' in modification:
            return self._apply_replace_modification(code, modification)

        elif 'insert_after' in modification and 'code' in modification:
            return self._apply_insert_after_modification(code, modification)

        elif 'insert_before' in modification and 'code' in modification:
            return self._apply_insert_before_modification(code, modification)

        return code

    def _apply_replace_modification(self, code, modification):
        """Apply replacement-type modification"""
        return code.replace(modification['find'], modification['replace'])

    def _apply_insert_after_modification(self, code, modification):
        """Apply insertion after specified content"""
        insert_pos = code.find(modification['insert_after'])
        if insert_pos != -1:
            insert_pos += len(modification['insert_after'])
            return code[:insert_pos] + '\n' + \
                modification['code'] + code[insert_pos:]
        return code

    def _apply_insert_before_modification(self, code, modification):
        """Apply insertion before specified content"""
        insert_pos = code.find(modification['insert_before'])
        if insert_pos != -1:
            return code[:insert_pos] + \
                modification['code'] + '\n' + code[insert_pos:]
        return code

    def _write_file_content(self, file_path, content):
        """Write file content"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _record_modification_history(
            self,
            file_path,
            original_code,
            modified_code,
            reason):
        """Record modification history"""
        self.code_history.append({
            'timestamp': time.time(),
            'type': 'modify',
            'file_path': file_path,
            'original_code': original_code,
            'modified_code': modified_code,
            'reason': reason
        })

    def improve_model_code(self, model_id, requirements=None):
        """Improve specified model's code based on requirements (refactored version)"""
        logger.info(
            f"Starting code improvement for model {model_id} (time: {time.strftime('%Y-%m-%d %H:%M:%S')})")
        try:
            # Validate model ID
            self._validate_model_id(model_id)

            # Get model environment information
            model_env = self._get_model_environment(model_id)

            # Get improvement requirements
            requirements = self._get_improvement_requirements(
                model_id, requirements)

            # Improve model files
            improvements = self._improve_model_files(
                model_id, model_env, requirements)

            return {
                'model_id': model_id,
                'improvements': improvements
            }
        except Exception as e:
            logger.error(f"Model code improvement error: {str(e)}")
            return {'error': str(e)}

    def _validate_model_id(self, model_id):
        """Validate if model ID exists"""
        if model_id not in self.model_environment_map:
            raise ValueError(f'Model does not exist: {model_id}')

    def _get_model_environment(self, model_id):
        """Get model environment information"""
        return self.model_environment_map[model_id]

    def _get_improvement_requirements(self, model_id, requirements):
        """Get improvement requirements (use default if not provided)"""
        if requirements is None:
            return [f"Optimize model {model_id} code"]
        return requirements

    def _improve_model_files(self, model_id, model_env, requirements):
        """Improve all related files of the model"""
        improvements = []

        # Improve main file
        improvements += self._improve_main_file(
            model_id, model_env, requirements)

        # Improve configuration files
        improvements += self._improve_config_files(
            model_id, model_env, requirements)

        return improvements

    def _improve_main_file(self, model_id, model_env, requirements):
        """Improve model's main file"""
        improvements = []
        if model_env['main_file']:
            improvements += self._improve_file(
                model_id,
                model_env['main_file'],
                requirements,
                "Main file"
            )
        return improvements

    def _improve_config_files(self, model_id, model_env, requirements):
        """Improve all configuration files"""
        improvements = []
        for config_file in model_env['config_files']:
            improvements += self._improve_file(
                model_id,
                config_file,
                requirements,
                "Configuration file"
            )
        return improvements

    def _start_self_improvement_monitor(self):
        """Start self-improvement monitoring thread
        """
        def monitor():
            while True:
                try:
                # Check every hour if self-improvement is needed
                    time.sleep(3600)
                    self._self_improve()
                except Exception as e:
                    logger.error(f"Self-improvement monitoring error: {str(e)}")
                    time.sleep(60)  # Wait 1 minute after error before retrying

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        logger.info("Self-improvement monitoring thread started")

    def _self_improve(self):
        """Perform self-improvement process
        """
        current_time = time.time()
        # Check if improvement interval is reached
        if current_time - self.last_self_improvement < self.self_improvement_interval:
            return
            
        try:
            # AGI Enhancement: Add knowledge base assisted self-improvement
            if self.knowledge_base:
                improvement_plan = self.knowledge_base.get_improvement_plan('K')
                if improvement_plan:
                    self.apply_knowledge_based_improvements(improvement_plan)
        except Exception as e:
            logger.error(f"Error during self-improvement process: {str(e)}")
        finally:
        # Ensure timestamp is updated regardless of success
            self.last_self_improvement = time.time()
                
    def apply_knowledge_based_improvements(self, improvement_plan: dict):
        """Apply improvements from knowledge base
        """
        logger.info(f"Applying knowledge base improvement plan: {improvement_plan['title']}")
        for step in improvement_plan['steps']:
            self.modify_code(
                step['file_path'],
                [step['modification']],
                reason=f"Knowledge base improvement: {step['reason']}"
            )

    def _save_improvement_suggestions(self, improvements):
        """Save improvement suggestions to file
        """
        try:
        # Use full path to ensure correct file location
            file_path = os.path.join(
                os.path.dirname(__file__),
                "programming_model_improvements.json")

        # Read existing data or initialize empty list
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
            else:
                existing_data = []

        # Add timestamp and append new improvement suggestions
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            improvements_with_time = {
                "timestamp": timestamp,
                "improvements": improvements
            }
            existing_data.append(improvements_with_time)

        # Write updated data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)

            logger.info(f"Improvement suggestions saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save improvement suggestions: {str(e)}")
            logger.error(traceback.format_exc())

    def _generate_self_improvement_suggestions(self):
        """Generate self-improvement suggestions
        """
        suggestions = []

        # 1. Check code complexity
        suggestions.append({
            "type": "Complexity Optimization",
            "description": "Analyze and refactor high complexity functions",
            "priority": "High"
        })

        # 2. Check dependency updates
        suggestions.append({
            "type": "Dependency Update",
            "description": "Check and update outdated dependencies",
            "priority": "Medium"
        })

        # 3. Performance optimization
        suggestions.append({
            "type": "Performance Optimization",
            "description": "Identify and optimize performance bottlenecks",
            "priority": "High"
        })

        # 4. Error handling enhancement
        suggestions.append({
            "type": "Error Handling",
            "description": "Enhance error handling and logging",
            "priority": "Medium"
        })

        return suggestions
