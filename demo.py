#!/usr/bin/env python3
"""
Demo script for the CoT Embeddings project.
Shows the complete pipeline in action with comparison between baseline and CoT approaches.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import load_config
from phase2.retrieval_pipeline import CoTRetrievalPipeline

def demo_retrieval_only():
    """Demo just the retrieval part (faster for testing)."""
    print("🔍 CoT Embeddings Retrieval Demo")
    print("=" * 50)
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Initialize retrieval pipeline
    print("Loading retrieval pipeline...")
    retrieval_pipeline = CoTRetrievalPipeline(config)
    retrieval_pipeline.load_components()
    
    # Test queries
    test_queries = [
        "What is 15 * 24?",
        "If I have 50 dollars and spend 18, how much is left?",
        "How do plants make their food?",
        "Why do objects fall when dropped?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 30)
        
        # Retrieve examples
        examples, scores = retrieval_pipeline.retrieve(query, k=2)
        
        # Show retrieved examples
        for j, (example, score) in enumerate(zip(examples, scores)):
            print(f"Retrieved Example {j+1} (similarity: {score:.4f}):")
            print(f"  Question: {example['question']}")
            print(f"  Rationale: {example['rationale'][:100]}...")
            print(f"  Answer: {example['answer']}")
            print()
        
        # Show how it would be formatted for generation
        prompt_format = retrieval_pipeline.retrieve_and_format(query, k=1, format_type="prompt")
        print("📋 Generated Prompt for CoT:")
        print(prompt_format)
        print("="*70)

def demo_baseline_vs_cot_simulation():
    """Simulate what the baseline vs CoT comparison would look like."""
    print("\n🎯 Baseline vs CoT Comparison (Simulated)")
    print("=" * 60)
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Initialize retrieval pipeline
    retrieval_pipeline = CoTRetrievalPipeline(config)
    retrieval_pipeline.load_components()
    
    test_queries = [
        "What is 15 * 24?",
        "If I have 50 dollars and spend 18, how much is left?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        
        # Simulate baseline approach
        print("📍 BASELINE (Direct Answer Only):")
        print(f"Prompt: 'Answer directly: {query}'")
        print("Expected Response: [Just the final answer, no reasoning]")
        print()
        
        # Show CoT approach
        print("🧠 RETRIEVAL-AUGMENTED CoT:")
        examples, scores = retrieval_pipeline.retrieve(query, k=1)
        if examples:
            best_example = examples[0]
            print(f"Retrieved Example (similarity: {scores[0]:.4f}):")
            print(f"  Question: {best_example['question']}")
            print(f"  Rationale: {best_example['rationale']}")
            print(f"  Answer: {best_example['answer']}")
            print()
            
            prompt_format = retrieval_pipeline.retrieve_and_format(query, k=1, format_type="prompt")
            print("Generated CoT Prompt:")
            print(prompt_format)
            print("Expected Response: [Step-by-step reasoning with final answer]")
        
        print("\n" + "="*70)

def main():
    """Run the demo."""
    print("🚀 CoT Embeddings Project Demo")
    print("This demo shows our three-phase pipeline:")
    print("  Phase 1: ✅ Built knowledge base with embeddings")
    print("  Phase 2: ✅ Retrieval pipeline working")
    print("  Phase 3: 🔄 Generation pipeline (ready for testing)")
    print()
    
    # Demo retrieval capabilities
    demo_retrieval_only()
    
    # Show comparison concept
    demo_baseline_vs_cot_simulation()
    
    print("\n🎉 Demo Complete!")
    print("\n📋 Next Steps:")
    print("1. The retrieval system successfully finds relevant reasoning examples")
    print("2. Examples are properly formatted for CoT generation")
    print("3. Ready to test with actual T5 model generation")
    print("4. Can compare baseline (direct answer) vs CoT (step-by-step) approaches")
    print("\nTo run full generation testing:")
    print("python src/phase3/generation_pipeline.py")

if __name__ == "__main__":
    main()