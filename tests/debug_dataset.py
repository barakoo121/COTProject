#!/usr/bin/env python3
"""
Debug script to understand the dataset loading issue.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import load_config
from phase1.dataset_processor import CoTDatasetProcessor

def debug_dataset():
    """Debug the dataset loading."""
    print("🔍 Debugging Dataset Loading")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    
    # Initialize processor
    processor = CoTDatasetProcessor(config)
    
    print("📥 Loading dataset (limited to 10 samples)...")
    try:
        # Load dataset with small limit for debugging
        dataset = processor.load_dataset(max_samples=10)
        print(f"Called load_dataset with max_samples=10")
        
        print(f"✅ Dataset loaded successfully")
        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset length: {len(dataset)}")
        
        if isinstance(dataset, dict):
            print(f"Keys (first 5): {list(dataset.keys())[:5]}")
            
            # Check first item
            first_key = next(iter(dataset.keys()))
            first_value = dataset[first_key]
            print(f"First key: {first_key}")
            print(f"First value type: {type(first_value)}")
            
            if isinstance(first_value, dict):
                print(f"First value keys: {list(first_value.keys())}")
                print(f"First value content:")
                for k, v in first_value.items():
                    if isinstance(v, str) and len(v) > 50:
                        print(f"  {k}: {v[:50]}...")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"First value: {first_value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset()