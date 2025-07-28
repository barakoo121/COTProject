#!/usr/bin/env python3
"""
Create synthetic test data for the CoT embeddings project.
This allows us to test the pipeline while the real dataset is downloading.
"""

import json
import random
from pathlib import Path

def create_synthetic_cot_data(num_samples: int = 100) -> list:
    """Create synthetic CoT data for testing."""
    
    # Sample question templates
    math_questions = [
        "What is {} + {}?",
        "If I have {} apples and give away {}, how many do I have left?",
        "Calculate {} * {}",
        "What is {} divided by {}?",
        "Find the area of a rectangle with length {} and width {}",
    ]
    
    reasoning_questions = [
        "Why do birds fly south for the winter?",
        "How does photosynthesis work in plants?",
        "What causes rain to form?",
        "Why do objects fall to the ground?",
        "How do computers process information?",
    ]
    
    # Sample rationale templates
    math_rationales = [
        "Let me solve this step by step. First, I need to add the numbers together. {} + {} = {}.",
        "I need to subtract the second number from the first. {} - {} = {}.",
        "To multiply these numbers, I'll use the multiplication table. {} × {} = {}.",
        "For division, I need to see how many times {} goes into {}. The answer is {}.",
        "To find the area, I multiply length by width. {} × {} = {}.",
    ]
    
    reasoning_rationales = [
        "This is a complex natural phenomenon. Let me think through the biological and environmental factors involved.",
        "This involves understanding the basic principles of biology and chemistry working together.",
        "This is related to the water cycle and atmospheric conditions that create precipitation.",
        "This is explained by the fundamental force of gravity acting on all objects with mass.",
        "This involves understanding how digital systems process and store information using binary code.",
    ]
    
    synthetic_data = []
    
    for i in range(num_samples):
        if i % 2 == 0:  # Math problems
            a, b = random.randint(1, 100), random.randint(1, 100)
            question = random.choice(math_questions).format(a, b)
            
            if "+" in question:
                answer = str(a + b)
                rationale = math_rationales[0].format(a, b, answer)
            elif "give away" in question or "-" in question:
                answer = str(max(0, a - b))
                rationale = math_rationales[1].format(a, b, answer)
            elif "*" in question:
                answer = str(a * b)
                rationale = math_rationales[2].format(a, b, answer)
            elif "divided" in question:
                answer = str(round(a / b, 2)) if b != 0 else "undefined"
                rationale = math_rationales[3].format(b, a, answer)
            else:  # Area
                answer = str(a * b)
                rationale = math_rationales[4].format(a, b, answer)
                
        else:  # Reasoning problems
            question = random.choice(reasoning_questions)
            rationale = random.choice(reasoning_rationales)
            
            # Generate appropriate answers
            if "birds" in question:
                answer = "Birds migrate south to find warmer climates and food sources during winter."
            elif "photosynthesis" in question:
                answer = "Plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
            elif "rain" in question:
                answer = "Water evaporates, forms clouds, and condenses to fall as precipitation."
            elif "fall" in question:
                answer = "Gravity pulls objects toward Earth's center."
            else:
                answer = "Computers use binary code to process and store information digitally."
        
        synthetic_data.append({
            "id": i,
            "question": question,
            "rationale": rationale,
            "answer": answer,
            "source": "synthetic_test_data"
        })
    
    return synthetic_data

def main():
    """Create and save synthetic test data."""
    
    # Create data directory
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    print("Creating synthetic CoT test data...")
    synthetic_data = create_synthetic_cot_data(100)
    
    # Save to JSON file
    output_path = data_dir / "synthetic_test_triplets.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(synthetic_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {len(synthetic_data)} synthetic examples")
    print(f"📁 Saved to: {output_path}")
    
    # Show first example
    print("\nFirst example:")
    print(json.dumps(synthetic_data[0], indent=2))
    
    # Show statistics
    print(f"\nStatistics:")
    print(f"- Total examples: {len(synthetic_data)}")
    print(f"- Average question length: {sum(len(d['question']) for d in synthetic_data) / len(synthetic_data):.1f} chars")
    print(f"- Average rationale length: {sum(len(d['rationale']) for d in synthetic_data) / len(synthetic_data):.1f} chars")
    print(f"- Average answer length: {sum(len(d['answer']) for d in synthetic_data) / len(synthetic_data):.1f} chars")

if __name__ == "__main__":
    main()