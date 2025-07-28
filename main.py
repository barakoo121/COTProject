#!/usr/bin/env python3
"""
CoT Embeddings Project - Main Entry Point

This script orchestrates the three-phase pipeline for Chain of Thought optimization:
1. Phase 1: Build and Index the Rationale Knowledge Base
2. Phase 2: The Retrieval Pipeline
3. Phase 3: The Generation Pipeline
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import load_config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cot_project.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main entry point for the CoT embeddings project."""
    parser = argparse.ArgumentParser(description="CoT Embeddings Optimization Project")
    parser.add_argument(
        "--phase", 
        type=int, 
        choices=[1, 2, 3], 
        help="Run specific phase (1: build index, 2: retrieval, 3: generation)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Test query for phases 2 and 3"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    try:
        if args.phase == 1:
            logger.info("Starting Phase 1: Building Knowledge Base")
            from phase1.embedding_generator import CoTEmbeddingGenerator
            from phase1.vector_indexer import CoTVectorIndexer
            
            # Generate embeddings
            generator = CoTEmbeddingGenerator(config)
            triplets = generator.load_triplets("synthetic_test_triplets.json")
            embeddings, metadata = generator.generate_embeddings(triplets)
            
            # Create vector index
            indexer = CoTVectorIndexer(config)
            indexer.create_index(embeddings, metadata)
            indexer.save_index()
            
            logger.info("Phase 1 completed successfully")
            
        elif args.phase == 2:
            logger.info("Starting Phase 2: Retrieval Pipeline")
            if not args.query:
                logger.error("--query required for Phase 2")
                return 1
                
            from phase2.retrieval_pipeline import CoTRetrievalPipeline
            
            retrieval_pipeline = CoTRetrievalPipeline(config)
            retrieval_pipeline.load_components()
            
            examples, scores = retrieval_pipeline.retrieve(args.query, k=3)
            
            print(f"\nQuery: {args.query}")
            print("Retrieved examples:")
            for i, (example, score) in enumerate(zip(examples, scores)):
                print(f"\n{i+1}. Similarity: {score:.4f}")
                print(f"   Question: {example['question']}")
                print(f"   Answer: {example['answer']}")
            
        elif args.phase == 3:
            logger.info("Starting Phase 3: Generation Pipeline")
            if not args.query:
                logger.error("--query required for Phase 3")
                return 1
                
            from phase3.generation_pipeline import CoTGenerationPipeline
            
            generation_pipeline = CoTGenerationPipeline(config)
            print("Loading models (this may take a while)...")
            generation_pipeline.load_components()
            
            print(f"\nComparing methods for: {args.query}")
            comparison = generation_pipeline.compare_methods(args.query)
            
            report = generation_pipeline.format_comparison_report(comparison)
            print(report)
            
        else:
            # Run demo
            logger.info("Running CoT Embeddings Demo")
            import subprocess
            subprocess.run([sys.executable, "demo.py"])
            
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        return 1
    
    logger.info("Execution completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())