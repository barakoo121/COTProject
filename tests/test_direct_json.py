#!/usr/bin/env python3
"""
Test script to directly process the CoT-Collection JSON file.
"""

import sys
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import load_config
from phase1.dataset_processor import CoTDatasetProcessor

def test_direct_json_processing():
    """Test processing the JSON file directly."""
    print("🔍 Testing Direct JSON Processing")
    print("=" * 50)
    
    # Download the JSON file
    print("📥 Downloading JSON file...")
    json_path = hf_hub_download(
        repo_id="kaist-ai/CoT-Collection",
        filename="data/CoT_collection_en.json",
        repo_type="dataset"
    )
    print(f"✅ Downloaded to: {json_path}")
    
    # Load configuration
    config = load_config()
    processor = CoTDatasetProcessor(config)
    
    # Load a small subset of the JSON data
    print("📖 Loading first 100 examples from JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total examples in dataset: {len(data)}")
    
    # Take first 100 examples
    sample_keys = list(data.keys())[:100]
    sample_data = {key: data[key] for key in sample_keys}
    
    print(f"Processing {len(sample_data)} examples...")
    
    # Process the sample data directly
    processed_triplets = []
    
    for idx, (key, example) in enumerate(sample_data.items()):
        try:
            triplet = processor._extract_triplet(example)
            if triplet:
                triplet['id'] = key
                triplet['source'] = 'cot_collection'
                processed_triplets.append(triplet)
                
        except Exception as e:
            print(f"Error processing example {key}: {e}")
            continue
    
    print(f"✅ Successfully processed {len(processed_triplets)} triplets")
    
    if processed_triplets:
        # Show first few examples
        print(f"\n📋 First 3 Processed Triplets:")
        for i, triplet in enumerate(processed_triplets[:3]):
            print(f"\nExample {i+1} (ID: {triplet['id']}):")
            print(f"  Task: {triplet.get('task', 'unknown')}")
            print(f"  Question: {triplet['question'][:100]}...")
            print(f"  Rationale: {triplet['rationale'][:100]}...")
            print(f"  Answer: {triplet['answer'][:50]}...")
        
        # Calculate statistics
        avg_q_len = sum(len(t['question']) for t in processed_triplets) / len(processed_triplets)
        avg_r_len = sum(len(t['rationale']) for t in processed_triplets) / len(processed_triplets)
        avg_a_len = sum(len(t['answer']) for t in processed_triplets) / len(processed_triplets)
        
        print(f"\n📈 Statistics:")
        print(f"  Success rate: {len(processed_triplets)}/{len(sample_data)} ({100*len(processed_triplets)/len(sample_data):.1f}%)")
        print(f"  Average question length: {avg_q_len:.1f} chars")
        print(f"  Average rationale length: {avg_r_len:.1f} chars")
        print(f"  Average answer length: {avg_a_len:.1f} chars")
        
        # Save sample
        output_path = Path("data/processed/real_cot_sample_100.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_triplets, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved sample to: {output_path}")
        
        return True
    else:
        print("❌ No triplets were successfully processed")
        return False

def main():
    """Run the test."""
    success = test_direct_json_processing()
    
    if success:
        print(f"\n🎉 Success! CoT-Collection data can be processed correctly.")
        print(f"\nNext steps:")
        print(f"• The real dataset structure is now working")
        print(f"• You can process larger subsets or the full dataset")
        print(f"• Use this data to create embeddings with real CoT examples")
    else:
        print(f"\n❌ Processing failed.")

if __name__ == "__main__":
    main()