"""
Generation pipeline for the CoT embeddings project.
Implements retrieval-augmented generation with CoT-T5 and comparison with baseline.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any
from openai import OpenAI
import time

logger = logging.getLogger(__name__)

class CoTGenerationPipeline:
    """Handles generation using retrieved examples and baseline comparison."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the generation pipeline.
        
        Args:
            config: Configuration dictionary containing generation settings
        """
        self.config = config
        self.generation_config = config['generation']
        
        # OpenAI configuration
        self.model_name = self.generation_config['model_name']
        self.max_tokens = self.generation_config['max_tokens']
        self.temperature = self.generation_config['temperature']
        
        # Initialize OpenAI client
        api_key = self.generation_config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or add api_key to config.")
        
        self.client = OpenAI(api_key=api_key)
        self.retrieval_pipeline = None
        
        # Cache for efficiency
        self._retrieval_loaded = False
    
    def load_components(self):
        """Load the retrieval pipeline (OpenAI client is already initialized)."""
        self._load_retrieval_pipeline()
    
    def _test_openai_connection(self):
        """Test OpenAI API connection."""
        try:
            # Make a simple test call to verify API key works
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0
            )
            logger.info("OpenAI API connection successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI API connection failed: {e}")
            raise
    
    def _load_retrieval_pipeline(self):
        """Load the retrieval pipeline."""
        if self._retrieval_loaded:
            return
        
        logger.info("Loading retrieval pipeline...")
        
        try:
            # Import here to avoid circular imports
            import sys
            sys.path.append(str(Path(__file__).parents[1]))
            from phase2.retrieval_pipeline import CoTRetrievalPipeline
            
            self.retrieval_pipeline = CoTRetrievalPipeline(self.config)
            self.retrieval_pipeline.load_components()
            self._retrieval_loaded = True
            logger.info("Retrieval pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load retrieval pipeline: {e}")
            raise
    
    def generate_baseline(self, query: str) -> Dict[str, Any]:
        """
        Generate response using baseline approach (no CoT, direct answer only).
        
        Args:
            query: Input query
            
        Returns:
            Dictionary with generation results
        """
        # Create baseline prompt that asks for direct answer only
        baseline_prompt = f"""Question: {query}

Give me the final answer only."""
        
        start_time = time.time()
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides direct, concise answers without showing reasoning steps."},
                    {"role": "user", "content": baseline_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            generated_text = response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            generated_text = f"Error: {str(e)}"
        
        generation_time = time.time() - start_time
        
        return {
            'method': 'baseline',
            'query': query,
            'prompt': baseline_prompt,
            'response': generated_text,
            'generation_time': generation_time,
            'retrieved_examples': None
        }
    
    def generate_with_cot(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Generate response using our retrieval-augmented CoT approach.
        
        Args:
            query: Input query
            k: Number of examples to retrieve
            
        Returns:
            Dictionary with generation results
        """
        if not self._retrieval_loaded:
            self._load_retrieval_pipeline()
        
        start_time = time.time()
        
        # Retrieve relevant examples
        examples, scores = self.retrieval_pipeline.retrieve(query, k=k)
        
        # Format the prompt with retrieved examples
        if examples:
            # Use up to 5 examples
            examples_text = ""
            for i, example in enumerate(examples[:5], 1):
                examples_text += f"""Example {i}:
Question: {example['question']}
Rationale: {example['rationale']}
Answer: {example['answer']}

"""
            
            cot_prompt = f"""Question: {query}

Here are examples of how to approach similar problems:

{examples_text}Now think step by step to solve the question above."""
        else:
            # Fallback if no examples retrieved
            cot_prompt = f"""Question: {query}

Think step by step to solve this question."""
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that thinks step-by-step and shows detailed reasoning before providing the final answer."},
                    {"role": "user", "content": cot_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            generated_text = response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            generated_text = f"Error: {str(e)}"
        
        generation_time = time.time() - start_time
        
        return {
            'method': 'retrieval_augmented_cot',
            'query': query,
            'prompt': cot_prompt,
            'response': generated_text,
            'generation_time': generation_time,
            'retrieved_examples': examples,
            'similarity_scores': scores
        }
    
    def compare_methods(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Compare baseline and CoT methods on the same query.
        
        Args:
            query: Input query
            k: Number of examples to retrieve for CoT method
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing methods for query: {query}")
        
        # Generate with both methods
        baseline_result = self.generate_baseline(query)
        cot_result = self.generate_with_cot(query, k=k)
        
        # Create comparison
        comparison = {
            'query': query,
            'baseline': baseline_result,
            'retrieval_augmented_cot': cot_result,
            'comparison_summary': {
                'baseline_time': baseline_result['generation_time'],
                'cot_time': cot_result['generation_time'],
                'time_difference': cot_result['generation_time'] - baseline_result['generation_time'],
                'retrieved_examples_count': len(cot_result['retrieved_examples']) if cot_result['retrieved_examples'] else 0,
                'best_similarity_score': cot_result['similarity_scores'][0] if cot_result['similarity_scores'] else None
            }
        }
        
        return comparison
    
    def batch_compare(self, queries: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """
        Compare methods on multiple queries.
        
        Args:
            queries: List of input queries
            k: Number of examples to retrieve for CoT method
            
        Returns:
            List of comparison results
        """
        results = []
        
        for query in queries:
            try:
                comparison = self.compare_methods(query, k=k)
                results.append(comparison)
            except Exception as e:
                logger.error(f"Failed to process query '{query}': {e}")
                results.append({
                    'query': query,
                    'error': str(e)
                })
        
        return results
    
    def evaluate_performance(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate performance metrics from comparison results.
        
        Args:
            comparisons: List of comparison results
            
        Returns:
            Performance evaluation metrics
        """
        if not comparisons:
            return {}
        
        # Filter out error cases
        valid_comparisons = [c for c in comparisons if 'error' not in c]
        
        if not valid_comparisons:
            return {'error': 'No valid comparisons to evaluate'}
        
        # Calculate metrics
        baseline_times = [c['comparison_summary']['baseline_time'] for c in valid_comparisons]
        cot_times = [c['comparison_summary']['cot_time'] for c in valid_comparisons]
        similarity_scores = [c['comparison_summary']['best_similarity_score'] for c in valid_comparisons if c['comparison_summary']['best_similarity_score'] is not None]
        
        metrics = {
            'total_queries': len(valid_comparisons),
            'average_baseline_time': sum(baseline_times) / len(baseline_times),
            'average_cot_time': sum(cot_times) / len(cot_times),
            'average_time_overhead': (sum(cot_times) - sum(baseline_times)) / len(baseline_times),
            'average_similarity_score': sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
            'retrieval_success_rate': len(similarity_scores) / len(valid_comparisons) if valid_comparisons else 0
        }
        
        return metrics
    
    def format_comparison_report(self, comparison: Dict[str, Any]) -> str:
        """
        Format a single comparison result into a readable report.
        
        Args:
            comparison: Comparison result dictionary
            
        Returns:
            Formatted report string
        """
        if 'error' in comparison:
            return f"Error processing query '{comparison['query']}': {comparison['error']}"
        
        report = []
        report.append("=" * 80)
        report.append(f"QUERY: {comparison['query']}")
        report.append("=" * 80)
        
        # Baseline results
        report.append("\n📍 BASELINE (Direct Answer Only):")
        report.append(f"Response: {comparison['baseline']['response']}")
        report.append(f"Generation time: {comparison['baseline']['generation_time']:.3f}s")
        
        # CoT results
        report.append("\n🧠 RETRIEVAL-AUGMENTED CoT:")
        if comparison['retrieval_augmented_cot']['retrieved_examples']:
            best_example = comparison['retrieval_augmented_cot']['retrieved_examples'][0]
            similarity = comparison['retrieval_augmented_cot']['similarity_scores'][0]
            report.append(f"Retrieved example (similarity: {similarity:.4f}):")
            report.append(f"  Question: {best_example['question']}")
            report.append(f"  Answer: {best_example['answer']}")
        
        report.append(f"Response: {comparison['retrieval_augmented_cot']['response']}")
        report.append(f"Generation time: {comparison['retrieval_augmented_cot']['generation_time']:.3f}s")
        
        # Summary
        summary = comparison['comparison_summary']
        report.append(f"\n📊 COMPARISON SUMMARY:")
        report.append(f"Time overhead: +{summary['time_difference']:.3f}s")
        report.append(f"Examples retrieved: {summary['retrieved_examples_count']}")
        if summary['best_similarity_score']:
            report.append(f"Best similarity: {summary['best_similarity_score']:.4f}")
        
        return "\n".join(report)

def main():
    """Test the generation pipeline with comparison."""
    import sys
    from pathlib import Path
    
    # Add src to path for imports
    sys.path.append(str(Path(__file__).parents[1]))
    from utils.config_loader import load_config
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Initialize generation pipeline
    generation_pipeline = CoTGenerationPipeline(config)
    
    # Load components
    print("Loading components...")
    generation_pipeline.load_components()
    
    # Test queries
    test_queries = [
        "What is 15 * 24?",
        "If I have 50 dollars and spend 18, how much is left?",
        "How do plants make their food?",
        "Why do objects fall when dropped?"
    ]
    
    print("Running comparisons...\n")
    
    # Run comparisons
    all_comparisons = []
    for query in test_queries:
        print(f"Processing: {query}")
        comparison = generation_pipeline.compare_methods(query, k=5)
        all_comparisons.append(comparison)
        
        # Print formatted report
        # report = generation_pipeline.format_comparison_report(comparison)
        # print(report)
        # print("\n" + "="*80 + "\n")
    
    # Evaluate overall performance
    # metrics = generation_pipeline.evaluate_performance(all_comparisons)
    # print("🎯 OVERALL PERFORMANCE METRICS:")
    # for key, value in metrics.items():
    #     if isinstance(value, float):
    #         print(f"{key}: {value:.4f}")
    #     else:
    #         print(f"{key}: {value}")

if __name__ == "__main__":
    main()