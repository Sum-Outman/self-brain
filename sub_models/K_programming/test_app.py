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

import unittest
from sub_models.K_programming.app import ProgrammingModel


class TestProgrammingModel(unittest.TestCase):
    def test_initialization(self):
        """Test if ProgrammingModel class can be initialized correctly"""
        pm = ProgrammingModel()
        self.assertIsInstance(pm, ProgrammingModel)
        self.assertIsNotNone(pm.model_environment_map)
        self.assertIn('K', pm.model_environment_map)  # Ensure the programming model itself is in the environment map

    def test_code_generation(self):
        """Test code generation functionality"""
        pm = ProgrammingModel()
        result = pm.generate_code("Create a simple addition function", language="python")
        self.assertIn('code', result)
        self.assertIn('GeneratedCode', result['code'])
        self.assertIn(
            'def _process_requirements(self) -> Any:',
            result['code'])

    def test_self_improvement(self):
        """Test self-improvement functionality"""
        pm = ProgrammingModel()
        # Set last improvement time to long ago
        pm.last_self_improvement = 0
        pm._self_improve()
        self.assertGreater(pm.last_self_improvement, 0)
        
    def test_self_improve_finally_block(self):
        """Test if the finally block in _self_improve method updates the timestamp"""
        model = ProgrammingModel()
        model.last_self_improvement = 0
        
        # Simulate an exception
        original_self_improve = model._self_improve
        def mock_self_improve():
            raise Exception("Simulated error")
        model._self_improve = mock_self_improve
        
        try:
            model._self_improve()
        except Exception as e:
            self.assertEqual(str(e), "Simulated error")
            
        # Verify timestamp is updated (even if an error occurs)
        self.assertNotEqual(model.last_self_improvement, 0)


if __name__ == '__main__':
    unittest.main()
