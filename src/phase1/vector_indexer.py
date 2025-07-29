"""
Vector indexer for the CoT embeddings project.
Creates and manages FAISS indices for fast similarity search.
"""

import logging
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import json

logger = logging.getLogger(__name__)

class CoTVectorIndexer:
    """Creates and manages FAISS vector index for CoT embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector indexer.
        
        Args:
            config: Configuration dictionary containing vector index settings
        """
        self.config = config
        self.vector_config = config['vector_index']
        self.index_path = Path(self.vector_config['index_path'])
        self.dimension = self.vector_config['dimension']
        self.index_type = self.vector_config['index_type']
        
        self.processed_dir = Path(config['dataset']['processed_dir'])
        
        # Ensure index directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.metadata = None
        self.id_to_index_map = {}
        self.index_to_id_map = {}
    
    def create_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> faiss.Index:
        """
        Create a FAISS index from embeddings.
        
        Args:
            embeddings: Array of embeddings to index
            metadata: Metadata for each embedding
            
        Returns:
            Created FAISS index
        """
        logger.info(f"Creating FAISS index with {len(embeddings)} embeddings")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"Index type: {self.index_type}")
        
        # Update dimension if different from config
        if embeddings.shape[1] != self.dimension:
            logger.warning(f"Updating dimension from {self.dimension} to {embeddings.shape[1]}")
            self.dimension = embeddings.shape[1]
        
        # Create the appropriate FAISS index
        if self.index_type == "IndexFlatIP":
            # Inner product index (for normalized embeddings, equivalent to cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            # L2 distance index
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # IVF (Inverted File) index for faster search on large datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = min(100, len(embeddings) // 10)  # Number of clusters, roughly sqrt(n)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Ensure embeddings are in the correct format
        embeddings = embeddings.astype(np.float32)
        
        # Train the index if necessary (for IVF indices)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training the index...")
            self.index.train(embeddings)
        
        # Add embeddings to the index
        logger.info("Adding embeddings to index...")
        self.index.add(embeddings)
        
        # Store metadata and create mappings
        self.metadata = metadata
        self.id_to_index_map = {item['id']: i for i, item in enumerate(metadata)}
        self.index_to_id_map = {i: item['id'] for i, item in enumerate(metadata)}
        
        logger.info(f"Index created successfully with {self.index.ntotal} vectors")
        
        return self.index
    
    def save_index(self, index_filename: str = "cot_faiss_index.bin") -> Path:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            index_filename: Name of the index file
            
        Returns:
            Path to the saved index file
        """
        if self.index is None:
            raise ValueError("No index to save. Create an index first.")
        
        # Save FAISS index
        index_file_path = self.index_path / index_filename
        faiss.write_index(self.index, str(index_file_path))
        
        # Save metadata and mappings
        metadata_path = self.index_path / "index_metadata.json"
        metadata_info = {
            'metadata': self.metadata,
            'id_to_index_map': self.id_to_index_map,
            'index_to_id_map': self.index_to_id_map,
            'index_info': {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'total_vectors': self.index.ntotal,
                'index_filename': index_filename
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved FAISS index to: {index_file_path}")
        logger.info(f"Saved metadata to: {metadata_path}")
        
        return index_file_path
    
    def load_index(self, index_filename: str = "cot_faiss_index.bin") -> faiss.Index:
        """
        Load a previously saved FAISS index from disk.
        
        Args:
            index_filename: Name of the index file
            
        Returns:
            Loaded FAISS index
        """
        index_file_path = self.index_path / index_filename
        metadata_path = self.index_path / "index_metadata.json"
        
        if not index_file_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_file_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_file_path))
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_info = json.load(f)
        
        self.metadata = metadata_info['metadata']
        self.id_to_index_map = {int(k): v for k, v in metadata_info['id_to_index_map'].items()}
        self.index_to_id_map = {int(k): v for k, v in metadata_info['index_to_id_map'].items()}
        
        index_info = metadata_info['index_info']
        self.dimension = index_info['dimension']
        
        logger.info(f"Loaded FAISS index from: {index_file_path}")
        logger.info(f"Index contains {self.index.ntotal} vectors")
        logger.info(f"Dimension: {self.dimension}")
        
        return self.index
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 5) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Search for the most similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (similarity_scores, retrieved_metadata)
        """
        if self.index is None:
            raise ValueError("No index loaded. Create or load an index first.")
        
        # Ensure query is in correct format
        query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, k)
        
        # Convert results
        scores = scores[0].tolist()  # First (and only) query
        indices = indices[0].tolist()
        
        # Get metadata for retrieved items
        retrieved_metadata = []
        for idx in indices:
            if idx < len(self.metadata):
                retrieved_metadata.append(self.metadata[idx])
            else:
                logger.warning(f"Index {idx} out of range for metadata")
        
        return scores, retrieved_metadata
    
    def get_item_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific item by its ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            Item metadata or None if not found
        """
        if item_id in self.id_to_index_map:
            index = self.id_to_index_map[item_id]
            return self.metadata[index]
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {}
        
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': getattr(self.index, 'is_trained', True),
            'num_metadata_items': len(self.metadata) if self.metadata else 0
        }
        
        return stats
    
    def rebuild_index(self, 
                     embeddings_filename: str = "question_embeddings.npy",
                     metadata_filename: str = "embedding_metadata.json"):
        """
        Rebuild the index from saved embeddings and metadata.
        
        Args:
            embeddings_filename: Name of the embeddings file
            metadata_filename: Name of the metadata file
        """
        # Load embeddings and metadata
        embeddings_path = self.processed_dir / embeddings_filename
        metadata_path = self.processed_dir / metadata_filename
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load data
        embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"Rebuilding index from {len(embeddings)} embeddings")
        
        # Create new index
        self.create_index(embeddings, metadata)
        
        # Save the rebuilt index
        self.save_index()

def main():
    """Test the vector indexer."""
    import sys
    from pathlib import Path
    
    # Add src to path for imports
    sys.path.append(str(Path(__file__).parents[1]))
    from utils.config_loader import load_config
    from phase1.embedding_generator import CoTEmbeddingGenerator
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Initialize indexer
    indexer = CoTVectorIndexer(config)
    
    # Load embeddings (assuming they were generated previously)
    try:
        # Use processed_dir from config for proper path resolution
        embeddings_path = Path(config['dataset']['processed_dir']) / "question_embeddings.npy"
        metadata_path = Path(config['dataset']['processed_dir']) / "embedding_metadata.json"
        
        embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded {len(embeddings)} embeddings")
        
    except FileNotFoundError:
        print("Embeddings not found. Generating them first...")
        
        raise Exception("Embeddings not found. Generate them first.")
    
    # Create index
    indexer.create_index(embeddings, metadata)
    
    # Save index
    indexer.save_index()
    
    # Test search
    if len(embeddings) > 0:
        # Use first embedding as a query
        query_embedding = embeddings[0]
        scores, results = indexer.search(query_embedding, k=5)
        
        print(f"\nTop 5 search results:")
        for i, (score, result) in enumerate(zip(scores, results)):
            print(f"  {i+1}. Score: {score:.4f}")
            print(f"     Question: {result['question'][:80]}...")
            print(f"     Answer: {result['answer'][:50]}...")
    
    # Get statistics
    stats = indexer.get_statistics()
    print(f"\nIndex statistics: {stats}")

if __name__ == "__main__":
    main()