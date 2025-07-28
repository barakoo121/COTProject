#!/usr/bin/env python3
"""
Test the CoT dataset processor with 5K pre-loaded samples.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import load_config
from phase1.dataset_processor import CoTDatasetProcessor

def test_5k_dataset():
    """Test with 5K pre-loaded samples."""
    print("🔍 Testing 5K CoT Dataset Processing")
    print("=" * 50)
    
    # Load the 5K dataset directly
    data_path = "data/processed/cot_5k_raw.json"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print("Run 'python test_direct_load.py' first to create the 5K dataset")
        return False
    
    print(f"📖 Loading 5K dataset from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset_dict = json.load(f)
    
    print(f"✅ Loaded {len(dataset_dict)} examples")
    print(f"First 5 keys: {list(dataset_dict.keys())[:5]}")
    
    # Now test processing
    config = load_config()
    processor = CoTDatasetProcessor(config)
    
    # Manually set the dataset
    processor.dataset = dataset_dict
    
    print(f"\n🔄 Processing {len(dataset_dict)} examples...")
    
    try:
        # Process the dataset
        triplets = processor.process_dataset()
        
        print(f"✅ Successfully processed {len(triplets)} triplets")
        success_rate = len(triplets) / len(dataset_dict) * 100
        print(f"Processing success rate: {success_rate:.1f}%")
        
        # Show first processed triplet
        if triplets:
            print(f"\n📋 First Processed Triplet:")
            first_triplet = triplets[0]
            for key, value in first_triplet.items():
                if isinstance(value, str) and len(value) > 80:
                    print(f"  {key}: {value[:80]}...")
                else:
                    print(f"  {key}: {value}")
        
        # Task distribution
        tasks = [t.get('task', 'unknown') for t in triplets]
        unique_tasks = set(tasks)
        print(f"\n📈 Task Distribution:")
        print(f"  Unique tasks: {len(unique_tasks)}")
        
        # Show top 10 most common tasks
        task_counts = {task: tasks.count(task) for task in unique_tasks}
        top_tasks = sorted(task_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for task, count in top_tasks:
            print(f"  {task}: {count} examples")
        
        # Save processed triplets
        output_path = processor.save_processed_data("cot_5k_processed.json")
        print(f"\n💾 Saved processed triplets to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    success = test_5k_dataset()
    
    if success:
        print(f"\n🎉 Success! 5K CoT examples processed successfully!")
        print(f"\nNext steps:")
        print(f"• Generate embeddings from 5K real examples")
        print(f"• Create vector index with diverse task coverage")
        print(f"• Test retrieval quality vs synthetic data")
        print(f"• Run baseline vs CoT comparison")
    else:
        print(f"\n❌ Test failed.")

if __name__ == "__main__":
    main()