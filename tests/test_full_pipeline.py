#!/usr/bin/env python3
"""
Test script to verify the complete CoT embeddings pipeline is working.
This script tests all three phases without requiring the heavy T5 model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import load_config
from phase1.embedding_generator import CoTEmbeddingGenerator
from phase1.vector_indexer import CoTVectorIndexer
from phase2.retrieval_pipeline import CoTRetrievalPipeline

def test_phase1():
    """Test Phase 1: Knowledge Base Creation"""
    print("🔧 Testing Phase 1: Knowledge Base Creation")
    print("-" * 50)
    
    config = load_config("config/config.yaml")
    
    # Test embedding generation
    generator = CoTEmbeddingGenerator(config)
    
    try:
        triplets = generator.load_triplets("synthetic_test_triplets.json")
        print(f"✅ Loaded {len(triplets)} synthetic triplets")
        
        embeddings, metadata = generator.generate_embeddings(triplets[:10])  # Test with 10 samples
        print(f"✅ Generated embeddings: {embeddings.shape}")
        
        # Test vector indexing
        indexer = CoTVectorIndexer(config)
        index = indexer.create_index(embeddings, metadata)
        print(f"✅ Created FAISS index with {index.ntotal} vectors")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        return False

def test_phase2():
    """Test Phase 2: Retrieval Pipeline"""
    print("\n🔍 Testing Phase 2: Retrieval Pipeline")
    print("-" * 50)
    
    config = load_config("config/config.yaml")
    
    try:
        # Initialize retrieval pipeline
        retrieval_pipeline = CoTRetrievalPipeline(config)
        retrieval_pipeline.load_components()
        print("✅ Loaded retrieval components")
        
        # Test retrieval
        test_queries = [
            "What is 5 * 6?",
            "How do plants grow?"
        ]
        
        for query in test_queries:
            examples, scores = retrieval_pipeline.retrieve(query, k=2)
            if examples and scores:
                print(f"✅ Retrieved {len(examples)} examples for: '{query}'")
                print(f"   Best similarity: {scores[0]:.4f}")
            else:
                print(f"❌ No examples retrieved for: '{query}'")
                return False
        
        # Test prompt formatting
        prompt = retrieval_pipeline.retrieve_and_format(
            "What is 10 + 15?", k=1, format_type="prompt"
        )
        if prompt and "step-by-step" in prompt:
            print("✅ Prompt formatting works correctly")
        else:
            print("❌ Prompt formatting failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        return False

def test_phase3_structure():
    """Test Phase 3: Generation Pipeline Structure (without actual generation)"""
    print("\n🧠 Testing Phase 3: Generation Pipeline Structure")
    print("-" * 50)
    
    try:
        from phase3.generation_pipeline import CoTGenerationPipeline
        
        config = load_config("config/config.yaml")
        generation_pipeline = CoTGenerationPipeline(config)
        print("✅ Generation pipeline initialized")
        
        # Test that the structure is correct without loading heavy models
        assert hasattr(generation_pipeline, 'generate_baseline')
        assert hasattr(generation_pipeline, 'generate_with_cot')
        assert hasattr(generation_pipeline, 'compare_methods')
        print("✅ All required methods present")
        
        # Test comparison structure (simulated)
        mock_comparison = {
            'query': 'test',
            'baseline': {'method': 'baseline', 'response': 'test', 'generation_time': 0.1, 'retrieved_examples': None},
            'retrieval_augmented_cot': {'method': 'cot', 'response': 'test', 'generation_time': 0.2, 'retrieved_examples': [{'question': 'test', 'answer': 'test'}], 'similarity_scores': [0.8]},
            'comparison_summary': {'baseline_time': 0.1, 'cot_time': 0.2, 'time_difference': 0.1, 'retrieved_examples_count': 1, 'best_similarity_score': 0.8}
        }
        
        report = generation_pipeline.format_comparison_report(mock_comparison)
        if "BASELINE" in report and "CoT" in report:
            print("✅ Comparison report formatting works")
        else:
            print("❌ Comparison report formatting failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 3 structure test failed: {e}")
        return False

def demonstrate_key_features():
    """Demonstrate the key features of our implementation."""
    print("\n🎯 Key Features Demonstration")
    print("=" * 60)
    
    config = load_config("config/config.yaml")
    retrieval_pipeline = CoTRetrievalPipeline(config)
    retrieval_pipeline.load_components()
    
    print("\n1. 📊 Semantic Similarity Matching:")
    print("   Query: 'What is 12 * 8?'")
    examples, scores = retrieval_pipeline.retrieve("What is 12 * 8?", k=2)
    for i, (example, score) in enumerate(zip(examples, scores)):
        print(f"   Match {i+1}: {example['question']} (similarity: {score:.4f})")
    
    print("\n2. 🔄 Baseline vs CoT Approach:")
    print("   Baseline Prompt: 'Answer directly: What is 12 * 8?'")
    print("   → Expected: Just '96'")
    print()
    print("   CoT Prompt: [Retrieved example] + 'Let's think step by step...'")
    print("   → Expected: 'To multiply 12 * 8, I need to... The answer is 96'")
    
    print("\n3. 📈 Performance Metrics:")
    print("   • Retrieval accuracy: Based on embedding similarity")
    print("   • Generation quality: CoT vs baseline reasoning")
    print("   • Time overhead: Retrieval + generation vs direct generation")
    
    print("\n4. ⚡ System Architecture:")
    print("   Phase 1: 768-dim embeddings → FAISS index")
    print("   Phase 2: Query embedding → Top-k retrieval")
    print("   Phase 3: Retrieved example → CoT prompt → T5 generation")

def main():
    """Run all tests."""
    print("🚀 CoT Embeddings Pipeline Testing")
    print("=" * 60)
    
    results = []
    
    # Test each phase
    results.append(("Phase 1", test_phase1()))
    results.append(("Phase 2", test_phase2()))
    results.append(("Phase 3 Structure", test_phase3_structure()))
    
    # Show results
    print("\n📋 Test Results Summary:")
    print("-" * 30)
    for phase, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{phase}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print(f"\n🎉 All tests passed! The pipeline is ready.")
        demonstrate_key_features()
        
        print("\n🚀 Next Steps:")
        print("• Run 'python demo.py' for full demonstration")
        print("• Run 'python main.py --phase 2 --query \"your question\"' to test retrieval")
        print("• Run 'python main.py --phase 3 --query \"your question\"' for full generation (requires T5 model)")
        print("• Replace synthetic data with real CoT-Collection when available")
        
    else:
        print(f"\n❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())