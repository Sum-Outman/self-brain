#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script to debug start_system.py
"""

import sys
import traceback
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        print("Attempting to import start_system...")
        import start_system
        print("Import successful!")
    except Exception as e:
        print(f"Error importing start_system: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)

    try:
        print("\nAttempting to call main() function...")
        if hasattr(start_system, 'main'):
            start_system.main()
        else:
            print("main() function not found in start_system")
    except Exception as e:
        print(f"Error calling main(): {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)