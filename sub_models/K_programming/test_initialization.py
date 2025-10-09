#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for programming model initialization
Tests direct initialization of ProgrammingModel without full UnifiedModelManager dependency
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_programming_model_initialization():
    """Test the initialization of the programming model"""
    try:
        logger.info("Starting programming model initialization test")
        
        # Import necessary modules
        from sub_models.K_programming.app import ProgrammingModel
        
        # Test if programming model class has initialize method
        if not hasattr(ProgrammingModel, 'initialize'):
            logger.error("Programming model class doesn't have initialize method")
            return False
        
        # Create a programming model instance
        programming_model = ProgrammingModel(language="python")
        logger.info("Created ProgrammingModel instance")
        
        # Test initialize method with a different mocking strategy since get_model_manager is dynamically imported
        import unittest.mock as mock
        
        # Create a magic mock for get_model_manager
        mock_manager = mock.MagicMock()
        mock_manager.get_model.return_value = mock.MagicMock()  # Mock knowledge base model
        
        # Create a simple function that returns our mock manager
        def mock_get_model_manager(*args, **kwargs):
            return mock_manager
        
        # Test the initialize method by manually patching the import
        # First, create a mock module for unified_model_manager
        mock_unified_module = mock.MagicMock()
        mock_unified_module.get_model_manager = mock_get_model_manager
        
        # Temporarily add the mock module to sys.modules
        original_module = sys.modules.get('unified_model_manager')
        sys.modules['unified_model_manager'] = mock_unified_module
        
        try:
            # Now call initialize
            result = programming_model.initialize()
            
            # Print the initialization result
            logger.info(f"Initialization result: {result}")
            
            if not result or (isinstance(result, dict) and not result.get('success', True)):
                logger.error(f"Programming model initialization test FAILED: {result.get('error', 'Unknown error')}")
                return False
            
            logger.info("Programming model initialization test PASSED")
            return True
        finally:
            # Restore the original module if it existed
            if original_module:
                sys.modules['unified_model_manager'] = original_module
            else:
                sys.modules.pop('unified_model_manager', None)
        
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    if test_programming_model_initialization():
        logger.info("Test passed! Programming model initialization implementation is correct")
    else:
        logger.error("Test failed! Please check the logs for details")