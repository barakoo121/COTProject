#!/usr/bin/env python3
"""
Test script to load and process the real CoT-Collection dataset.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import load_config
from phase1.dataset_processor import CoTDatasetProcessor

def test_real_dataset():
    """Test loading the real CoT-Collection dataset."""
    print("🔍 Testing Real CoT-Collection Dataset Loading")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize processor
    processor = CoTDatasetProcessor(config)
    
    print("📥 Loading dataset (this may take a while)...")
    try:
        # Load dataset with limited samples for testing
        dataset = processor.load_dataset(max_samples=1000)
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
        
        # Process a small subset
        print(f"\n🔄 Processing triplets...")
        triplets = processor.process_dataset()
        
        if triplets:
            print(f"✅ Successfully processed {len(triplets)} triplets")
            
            # Show first processed triplet
            print(f"\n📋 First Processed Triplet:")
            first_triplet = triplets[0]
            for key, value in first_triplet.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
            
            # Save processed data
            output_path = processor.save_processed_data("real_cot_triplets_sample.json")
            print(f"💾 Saved processed data to: {output_path}")
            
            # Show statistics
            stats = processor.get_statistics()
            print(f"\n📈 Processing Statistics:")
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
    print("🚀 Real CoT-Collection Dataset Test")
    print("This test loads and processes the actual 1.8M sample dataset")
    print("(Limited to 1000 samples for testing)\n")
    
    success = test_real_dataset()
    
    if success:
        print(f"\n🎉 Success! Real CoT-Collection dataset can be loaded and processed.")
        print(f"\nNext steps:")
        print(f"• Use the processed data to generate embeddings with real examples")
        print(f"• Compare retrieval quality between synthetic and real data")
        print(f"• Scale up to process the full 1.8M dataset")
    else:
        print(f"\n❌ Test failed. Check the error messages above.")

if __name__ == "__main__":
    main()