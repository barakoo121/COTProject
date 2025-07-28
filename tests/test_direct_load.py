#!/usr/bin/env python3
"""
Direct test to load 5K samples from CoT-Collection JSON.
"""

from huggingface_hub import hf_hub_download
import json

def test_direct_load():
    """Test loading 5K samples directly."""
    print("🔍 Direct Load Test (5K samples)")
    print("=" * 40)
    
    # Download JSON file
    print("📥 Downloading JSON file...")
    json_path = hf_hub_download(
        repo_id="kaist-ai/CoT-Collection",
        filename="data/CoT_collection_en.json",
        repo_type="dataset"
    )
    print(f"Downloaded to: {json_path}")
    
    # Load and limit data
    print("📖 Loading JSON data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    print(f"Total examples: {len(full_data)}")
    print(f"First 5 keys: {list(full_data.keys())[:5]}")
    
    # Limit to 5K samples
    max_samples = 5000
    all_keys = list(full_data.keys())
    limited_keys = all_keys[:max_samples]
    limited_data = {key: full_data[key] for key in limited_keys}
    
    print(f"Limited to {len(limited_data)} examples")
    print(f"Limited keys (first 5): {list(limited_data.keys())[:5]}")
    
    # Check structure of first example
    first_key = limited_keys[0]
    first_example = limited_data[first_key]
    
    print(f"\nFirst example (key: {first_key}):")
    print(f"  Type: {type(first_example)}")
    print(f"  Keys: {list(first_example.keys())}")
    
    for key, value in first_example.items():
        if isinstance(value, str) and len(value) > 50:
            print(f"  {key}: {value[:50]}...")
        else:
            print(f"  {key}: {value}")
    
    # Save the limited dataset
    output_path = "data/processed/cot_5k_raw.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(limited_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved 5K examples to: {output_path}")
    
    return limited_data

if __name__ == "__main__":
    result = test_direct_load()
    print(f"\n✅ Successfully loaded {len(result)} examples directly!")