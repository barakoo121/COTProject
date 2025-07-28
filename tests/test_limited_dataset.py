#!/usr/bin/env python3
"""
Test script to load and process CoT-Collection with limited samples (5K).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import load_config
from phase1.dataset_processor import CoTDatasetProcessor

def test_limited_dataset():
    """Test loading CoT-Collection with 5K samples limit."""
    print("🔍 Testing Limited CoT-Collection Dataset (5K samples)")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize processor
    processor = CoTDatasetProcessor(config)
    
    print("📥 Loading dataset (limited to 5K samples)...")
    try:
        # Load dataset with 5K samples limit
        dataset = processor.load_dataset(max_samples=5000)
        print(f"✅ Successfully loaded dataset")
        
        # Explore structure
        structure = processor.explore_dataset_structure()
        print(f"\n📊 Dataset Structure:")
        print(f"  Total samples: {structure['total_samples']}")
        print(f"  Columns: {structure['columns']}")
        
        # Show first example
        if structure['sample_data']:
            print(f"\n📝 First Example:")
            example = structure['sample_data'][0]
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
        # Process the dataset
        print(f"\n🔄 Processing 5K triplets...")
        triplets = processor.process_dataset()
        
        if triplets:
            print(f"✅ Successfully processed {len(triplets)} triplets")
            
            # Show processing statistics
            success_rate = len(triplets) / structure['total_samples'] * 100
            print(f"Processing success rate: {success_rate:.1f}%")
            
            # Show first processed triplet
            print(f"\n📋 First Processed Triplet:")
            first_triplet = triplets[0]
            for key, value in first_triplet.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
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
            
            # Save processed data
            output_path = processor.save_processed_data("cot_5k_triplets.json")
            print(f"\n💾 Saved processed data to: {output_path}")
            
            # Show statistics
            stats = processor.get_statistics()
            print(f"\n📊 Processing Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            
            return True
        else:
            print("❌ No triplets were processed")
            return False
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("🚀 Limited CoT-Collection Dataset Test")
    print("This test loads 5K samples instead of the full 1.8M dataset\n")
    
    success = test_limited_dataset()
    
    if success:
        print(f"\n🎉 Success! CoT-Collection with 5K samples loaded and processed.")
        print(f"\nBenefits of limited loading:")
        print(f"• ✅ Much faster loading time")
        print(f"• ✅ Still diverse task coverage")
        print(f"• ✅ Sufficient for testing and development")
        print(f"• ✅ Can scale up when needed")
        print(f"\nNext steps:")
        print(f"• Generate embeddings from 5K real examples")
        print(f"• Test retrieval quality with diverse tasks")
        print(f"• Run baseline vs CoT comparison")
    else:
        print(f"\n❌ Test failed. Check the error messages above.")

if __name__ == "__main__":
    main()