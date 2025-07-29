"""
Retrieval pipeline for the CoT embeddings project.
Handles query processing and retrieval of relevant reasoning examples.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class CoTRetrievalPipeline:
    """Handles retrieval of relevant reasoning examples for new queries."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the retrieval pipeline.
        
        Args:
            config: Configuration dictionary containing retrieval settings
        """
        self.config = config
        self.embedding_config = config['embedding']
        self.retrieval_config = config['retrieval']
        self.vector_config = config['vector_index']
        
        # Configuration parameters
        self.model_name = self.embedding_config['model_name']
        self.top_k = self.retrieval_config['top_k']
        self.similarity_threshold = self.retrieval_config['similarity_threshold']
        self.device = config['system']['device']
        
        # Paths
        self.processed_dir = Path(config['dataset']['processed_dir'])
        self.index_path = Path(self.vector_config['index_path'])
        
        # Components
        self.embedding_model = None
        self.vector_indexer = None
        
        # Cache for efficiency
        self._model_loaded = False
        self._index_loaded = False
    
    def load_components(self):
        """Load the embedding model and vector index."""
        self._load_embedding_model()
        self._load_vector_index()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for query encoding."""
        if self._model_loaded:
            return
        
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            self._model_loaded = True
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_vector_index(self):
        """Load the FAISS vector index."""
        if self._index_loaded:
            return
        
        logger.info("Loading vector index...")
        
        try:
            # Import here to avoid circular imports
            import sys
            sys.path.append(str(Path(__file__).parents[1]))
            from phase1.vector_indexer import CoTVectorIndexer
            
            self.vector_indexer = CoTVectorIndexer(self.config)
            self.vector_indexer.load_index()
            self._index_loaded = True
            logger.info("Vector index loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query string into an embedding vector.
        
        Args:
            query: Query string to encode
            
        Returns:
            Query embedding vector
        """
        if not self._model_loaded:
            self._load_embedding_model()
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_tensor=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return query_embedding[0]  # Return single embedding
    
    def retrieve(self, 
                query: str, 
                k: Optional[int] = None,
                return_scores: bool = True) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve the most relevant reasoning examples for a query.
        
        Args:
            query: Query string
            k: Number of examples to retrieve (uses config default if None)
            return_scores: Whether to return similarity scores
            
        Returns:
            Tuple of (retrieved_examples, similarity_scores)
        """
        if not self._index_loaded:
            self._load_vector_index()
        
        if k is None:
            k = self.top_k
        
        print(f"\n🔍 SEARCHING FOR SIMILAR QUESTIONS")
        print(f"Query: {query}")
        print(f"Looking for top {k} similar examples...")
        print("-" * 80)
        
        logger.info(f"Retrieving {k} examples for query: {query[:100]}...")
        
        # Encode the query
        query_embedding = self.encode_query(query)
        
        # Search the vector index
        scores, retrieved_metadata = self.vector_indexer.search(query_embedding, k=k)
        
        # Print the retrieved examples
        print(f"📋 RETRIEVED {len(retrieved_metadata)} SIMILAR EXAMPLES:")
        for i, (metadata, score) in enumerate(zip(retrieved_metadata, scores), 1):
            print(f"\n{i}. Similarity Score: {score:.4f}")
            print(f"   Question: {metadata['question']}")
            print(f"   Answer: {metadata['answer']}")
            if len(metadata['rationale']) > 100:
                print(f"   Rationale: {metadata['rationale'][:100]}...")
            else:
                print(f"   Rationale: {metadata['rationale']}")
        print("-" * 80)
        
        # Filter by similarity threshold if specified
        if self.similarity_threshold > 0:
            filtered_results = []
            filtered_scores = []
            
            for score, metadata in zip(scores, retrieved_metadata):
                if score >= self.similarity_threshold:
                    filtered_results.append(metadata)
                    filtered_scores.append(score)
                else:
                    logger.debug(f"Filtered out result with score {score:.4f} (below threshold {self.similarity_threshold})")
            
            if len(filtered_results) == 0:
                logger.warning(f"No results above similarity threshold {self.similarity_threshold}")
                # Return top result anyway
                filtered_results = retrieved_metadata[:1]
                filtered_scores = scores[:1]
            
            retrieved_metadata = filtered_results
            scores = filtered_scores
        
        logger.info(f"Retrieved {len(retrieved_metadata)} examples")
        
        if return_scores:
            return retrieved_metadata, scores
        else:
            return retrieved_metadata, []
    
    def retrieve_and_format(self, 
                           query: str, 
                           k: Optional[int] = None,
                           format_type: str = "simple") -> str:
        """
        Retrieve examples and format them for downstream use.
        
        Args:
            query: Query string
            k: Number of examples to retrieve
            format_type: Format type ("simple", "detailed", "prompt")
            
        Returns:
            Formatted string with retrieved examples
        """
        # For prompt format, ensure we get at least 5 examples
        if format_type == "prompt" and (k is None or k < 5):
            k = 5
            
        retrieved_examples, scores = self.retrieve(query, k=k)
        
        if format_type == "simple":
            return self._format_simple(retrieved_examples, scores)
        elif format_type == "detailed":
            return self._format_detailed(retrieved_examples, scores)
        elif format_type == "prompt":
            return self._format_for_prompt(retrieved_examples, scores)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def _format_simple(self, examples: List[Dict[str, Any]], scores: List[float]) -> str:
        """Format examples in a simple text format."""
        formatted = []
        
        for i, (example, score) in enumerate(zip(examples, scores)):
            formatted.append(f"Example {i+1} (similarity: {score:.4f}):")
            formatted.append(f"  Question: {example['question']}")
            formatted.append(f"  Rationale: {example['rationale']}")
            formatted.append(f"  Answer: {example['answer']}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_detailed(self, examples: List[Dict[str, Any]], scores: List[float]) -> str:
        """Format examples with detailed metadata."""
        formatted = []
        
        for i, (example, score) in enumerate(zip(examples, scores)):
            formatted.append(f"=== Retrieved Example {i+1} ===")
            formatted.append(f"Similarity Score: {score:.4f}")
            formatted.append(f"Source: {example.get('source', 'unknown')}")
            formatted.append(f"ID: {example.get('id', 'unknown')}")
            formatted.append(f"Question: {example['question']}")
            formatted.append(f"Rationale: {example['rationale']}")
            formatted.append(f"Answer: {example['answer']}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_for_prompt(self, examples: List[Dict[str, Any]], scores: List[float]) -> str:
        """Format examples for use in generation prompts (Phase 3)."""
        if not examples:
            return ""
        
        # Use top 5 examples for the prompt format
        top_examples = examples[:5]  # Get up to 5 examples
        
        formatted_examples = []
        for i, example in enumerate(top_examples):
            formatted_examples.append(f"Example {i+1}:")
            formatted_examples.append(f"Question: {example['question']}")
            formatted_examples.append(f"Rationale: {example['rationale']}")
            formatted_examples.append("")  # Empty line between examples
        
        prompt_template = """Here are examples demonstrating how to think step-by-step:

{examples}---

Now, solve the following problem using the same step-by-step thinking:"""
        
        return prompt_template.format(
            examples="\n".join(formatted_examples)
        )

def main():
    """Test the retrieval pipeline."""
    import sys
    from pathlib import Path
    
    # Add src to path for imports
    sys.path.append(str(Path(__file__).parents[1]))
    from utils.config_loader import load_config
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Initialize retrieval pipeline
    retrieval_pipeline = CoTRetrievalPipeline(config)
    
    # Load components
    retrieval_pipeline.load_components()
    
    # Test queries
    test_queries = [
        "What is 25 * 17?",
        "How do plants make food?",
        "If I have 100 dollars and spend 30, how much is left?",
        "Why do things fall down?"
    ]
    
    print("Testing retrieval pipeline...\n")
    
    for i, query in enumerate(test_queries):
        print(f"=== Query {i+1}: {query} ===")
        
        # Retrieve examples
        retrieval_pipeline.retrieve(query, k=5)

        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()