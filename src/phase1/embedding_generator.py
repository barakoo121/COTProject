"""
Embedding generator for the CoT embeddings project.
Generates sentence embeddings for questions using sentence-transformers.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

class CoTEmbeddingGenerator:
    """Generates embeddings for questions in the CoT data§set."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding generator.
        
        Args:
            config: Configuration dictionary containing embedding settings
        """
        self.config = config
        self.embedding_config = config['embedding']
        self.model_name = self.embedding_config['model_name']
        self.batch_size = self.embedding_config['batch_size']
        self.max_length = self.embedding_config['max_length']
        
        self.processed_dir = Path(config['dataset']['processed_dir'])
        self.device = config['system']['device']
        
        # Initialize the sentence transformer model
        self.model = None
        self.embeddings = None
        self.metadata = None
    
    def load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_triplets(self, filename: str = "synthetic_test_triplets.json") -> List[Dict[str, Any]]:
        """
        Load processed triplets from disk.
        
        Args:
            filename: Name of the file containing processed triplets
            
        Returns:
            List of triplets
        """
        triplet_path = self.processed_dir / filename
        
        if not triplet_path.exists():
            raise FileNotFoundError(f"Triplets file not found: {triplet_path}")
        
        with open(triplet_path, 'r', encoding='utf-8') as f:
            triplets = json.load(f)
        
        logger.info(f"Loaded {len(triplets)} triplets from {triplet_path}")
        return triplets
    
    def generate_embeddings(self, 
                          triplets: List[Dict[str, Any]], 
                          save_embeddings: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generate embeddings for all questions in the triplets.
        
        Args:
            triplets: List of question-rationale-answer triplets
            save_embeddings: Whether to save embeddings to disk
            
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Generating embeddings for {len(triplets)} questions...")
        
        # Extract questions
        questions = [triplet['question'] for triplet in triplets]
        
        # Generate embeddings in batches
        all_embeddings = []
        
        for i in tqdm(range(0, len(questions), self.batch_size), desc="Generating embeddings"):
            batch_questions = questions[i:i + self.batch_size]
            
            # Generate embeddings for this batch
            batch_embeddings = self.model.encode(
                batch_questions,
                batch_size=self.batch_size,
                convert_to_tensor=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=False
            )
            
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Create metadata for each embedding
        metadata = []
        for i, triplet in enumerate(triplets):
            metadata.append({
                'id': triplet.get('id', i),
                'question': triplet['question'],
                'rationale': triplet['rationale'],
                'answer': triplet['answer'],
                'source': triplet.get('source', 'unknown'),
                'embedding_index': i
            })
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Store embeddings and metadata
        self.embeddings = embeddings
        self.metadata = metadata
        
        # Save to disk if requested
        if save_embeddings:
            self.save_embeddings()
        
        return embeddings, metadata
    
    def save_embeddings(self, 
                       embeddings_filename: str = "question_embeddings.npy",
                       metadata_filename: str = "embedding_metadata.json"):
        """
        Save embeddings and metadata to disk.
        
        Args:
            embeddings_filename: Name of the embeddings file
            metadata_filename: Name of the metadata file
        """
        if self.embeddings is None or self.metadata is None:
            raise ValueError("No embeddings to save. Generate embeddings first.")
        
        # Save embeddings as numpy array
        embeddings_path = self.processed_dir / embeddings_filename
        np.save(embeddings_path, self.embeddings)
        
        # Save metadata as JSON
        metadata_path = self.processed_dir / metadata_filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved embeddings to: {embeddings_path}")
        logger.info(f"Saved metadata to: {metadata_path}")
        
        # Also save embedding configuration for reproducibility
        config_path = self.processed_dir / "embedding_config.json"
        embedding_info = {
            'model_name': self.model_name,
            'embedding_dimension': self.embeddings.shape[1],
            'num_embeddings': self.embeddings.shape[0],
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'device': self.device,
            'normalized': True
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_info, f, indent=2)
        
        logger.info(f"Saved embedding config to: {config_path}")
    
    def load_embeddings(self,
                       embeddings_filename: str = "question_embeddings.npy",
                       metadata_filename: str = "embedding_metadata.json") -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Load previously generated embeddings from disk.
        
        Args:
            embeddings_filename: Name of the embeddings file
            metadata_filename: Name of the metadata file
            
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        embeddings_path = self.processed_dir / embeddings_filename
        metadata_path = self.processed_dir / metadata_filename
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load embeddings
        embeddings = np.load(embeddings_path)
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
        logger.info(f"Loaded {len(metadata)} metadata entries")
        
        self.embeddings = embeddings
        self.metadata = metadata
        
        return embeddings, metadata
    
    def compute_similarity(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Compute similarity between a query embedding and all stored embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top similar embeddings to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Load or generate embeddings first.")
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the generated embeddings.
        
        Returns:
            Dictionary with embedding statistics
        """
        if self.embeddings is None:
            return {}
        
        stats = {
            'num_embeddings': self.embeddings.shape[0],
            'embedding_dimension': self.embeddings.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(self.embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(self.embeddings, axis=1))),
            'model_name': self.model_name,
            'device': self.device
        }
        
        return stats

def main():
    """Test the embedding generator."""
    import sys
    from pathlib import Path
    
    # Add src to path for config import
    sys.path.append(str(Path(__file__).parents[1]))
    from utils.config_loader import load_config
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Initialize generator
    generator = CoTEmbeddingGenerator(config)
    
    # Load test triplets
    triplets = generator.load_triplets("processed_triplets.json")
    
    # Generate embeddings
    embeddings, metadata = generator.generate_embeddings(triplets)
    
    # Get statistics
    stats = generator.get_statistics()
    print(f"\nEmbedding statistics: {stats}")
    
    # Test similarity computation
    if len(embeddings) > 0:
        # Use first embedding as query
        query_embedding = embeddings[0]
        similar_items = generator.compute_similarity(query_embedding, top_k=3)
        
        print(f"\nTop 3 similar items to first question:")
        for idx, similarity in similar_items:
            print(f"  Index {idx}: {similarity:.4f} - {metadata[idx]['question'][:50]}...")

if __name__ == "__main__":
    main()