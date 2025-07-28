#!/usr/bin/env python3
"""
Simple test script to explore the CoT-Collection dataset structure.
"""

import logging
from datasets import load_dataset
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_loading():
    """Test different methods to load the CoT-Collection dataset."""
    
    # Method 1: Load JSON file directly
    print("Method 1: Loading JSON file directly...")
    try:
        dataset = load_dataset(
            "json",
            data_files="hf://datasets/kaist-ai/CoT-Collection/data/CoT_collection_en.json"
        )
        print("✅ Successfully loaded with JSON method")
        print(f"Dataset: {dataset}")
        if 'train' in dataset:
            print(f"Train split size: {len(dataset['train'])}")
            print(f"Columns: {dataset['train'].column_names}")
            
            # Show first example
            first_example = dataset['train'][0]
            print("First example:")
            print(json.dumps(first_example, indent=2, default=str))
            
        return dataset
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Try loading a smaller subset for testing
    print("\nMethod 2: Loading with streaming to peek at structure...")
    try:
        from huggingface_hub import hf_hub_download
        import json as json_lib
        
        # Download the JSON file directly
        json_path = hf_hub_download(
            repo_id="kaist-ai/CoT-Collection",
            filename="data/CoT_collection_en.json",
            repo_type="dataset"
        )
        
        print(f"Downloaded JSON file to: {json_path}")
        
        # Read first few lines to understand structure
        with open(json_path, 'r', encoding='utf-8') as f:
            # Read first few lines
            examples = []
            for i, line in enumerate(f):
                if i >= 3:  # Read first 3 examples
                    break
                try:
                    example = json_lib.loads(line.strip())
                    examples.append(example)
                except:
                    continue
        
        print(f"✅ Successfully loaded {len(examples)} examples from JSON")
        print("First example:")
        print(json.dumps(examples[0], indent=2, default=str))
        
        return examples
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    print("❌ All methods failed")
    return None

if __name__ == "__main__":
    result = test_dataset_loading()