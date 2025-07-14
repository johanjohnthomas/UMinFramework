#!/usr/bin/env python3
"""Debug script to isolate the import causing segmentation fault"""

print("Starting import debugging...")

try:
    print("1. Testing basic imports...")
    import sys
    import os
    print("   Basic imports OK")
    
    print("2. Testing transformers import...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("   Transformers import OK")
    
    print("3. Testing luh components individually...")
    
    print("   3a. Importing luh module...")
    import luh
    print("   luh module import OK")
    
    print("   3b. Importing AutoUncertaintyHead...")
    from luh import AutoUncertaintyHead
    print("   AutoUncertaintyHead import OK")
    
    print("   3c. Importing CausalLMWithUncertainty...")
    from luh import CausalLMWithUncertainty
    print("   CausalLMWithUncertainty import OK")
    
    print("All imports successful!")
    
except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other error: {e}")
    import traceback
    traceback.print_exc()
