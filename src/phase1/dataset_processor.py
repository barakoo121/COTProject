"""
Dataset processor for the CoT-Collection dataset.
Loads and processes the 1.84M question-rationale pairs from kaist-ai/CoT-Collection.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
from datasets import Dataset
import json

logger = logging.getLogger(__name__)

class CoTDatasetProcessor:
    """Processes the CoT-Collection dataset for embedding and indexing."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset processor.
        
        Args:
            config: Configuration dictionary containing dataset settings
        """
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.cache_dir = Path(config['dataset']['cache_dir'])
        self.processed_dir = Path(config['dataset']['processed_dir'])
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset = None
        self.processed_data = None
    
    def load_dataset(self, subset: str = None, max_samples: int = 15000) -> Dataset:
        """
        Load the CoT-Collection dataset from §e.
        
        Args:
            subset: Specific subset to load (if any)
            max_samples: Maximum number of samples to load (for testing)
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            # Load the CoT-Collection dataset using the JSON file approach
            # The dataset has a JSON file at data/CoT_collection_en.json
            logger.info("Loading CoT-Collection dataset from JSON file...")
            
            # For large datasets like CoT-Collection, we need to limit the data loaded
            # Download the JSON file first and then load a subset
            try:
                from huggingface_hub import hf_hub_download
                import json
                
                logger.info("Downloading CoT-Collection JSON file...")
                json_path = hf_hub_download(
                    repo_id="kaist-ai/CoT-Collection",
                    filename="data/CoT_collection_en.json",
                    repo_type="dataset",
                    cache_dir=str(self.cache_dir)
                )
                logger.info(f"Downloaded JSON file to: {json_path}")
                
                # Load the JSON file and limit the number of examples
                logger.info("Loading and limiting dataset size...")
                with open(json_path, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                
                total_examples = len(full_data)
                logger.info(f"Total examples in dataset: {total_examples}")
                logger.info(f"First 5 keys from full_data: {list(full_data.keys())[:5]}")
                
                # Limit the number of examples if max_samples is specified
                if max_samples and max_samples < total_examples:
                    # Take the first max_samples examples
                    all_keys = list(full_data.keys())
                    limited_keys = all_keys[:max_samples]
                    limited_data = {key: full_data[key] for key in limited_keys}
                    logger.info(f"Limited dataset to {max_samples} examples from {total_examples} total")
                    logger.info(f"Limited keys: {limited_keys}")
                else:
                    limited_data = full_data
                    logger.info(f"Using all {total_examples} examples")
                
                # Store the data directly (since it's already a dictionary)
                self.dataset = limited_data
                logger.info(f"Successfully loaded CoT-Collection dataset with {len(self.dataset)} examples")
                logger.info(f"Final dataset keys (first 5): {list(self.dataset.keys())[:5]}")
                
            except Exception as e:
                logger.error(f"Failed to load CoT-Collection dataset: {e}")
                raise RuntimeError(f"Could not load CoT-Collection dataset: {e}")
            
            logger.info(f"Loaded {len(self.dataset)} samples")
            
            # Show columns/keys information
            if isinstance(self.dataset, dict):
                if len(self.dataset) > 0:
                    first_key = next(iter(self.dataset.keys()))
                    first_value = self.dataset[first_key]
                    if isinstance(first_value, dict):
                        sample_keys = list(first_value.keys())
                        logger.info(f"Dataset keys: {sample_keys}")
                    else:
                        logger.info(f"Dataset structure: Dictionary with {len(self.dataset)} entries")
            else:
                if hasattr(self.dataset, 'column_names'):
                    logger.info(f"Dataset columns: {self.dataset.column_names}")
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def explore_dataset_structure(self) -> Dict[str, Any]:
        """
        Explore the dataset structure to understand the data format.
        
        Returns:
            Dictionary with dataset structure information
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Handle dictionary format (CoT-Collection)
        if isinstance(self.dataset, dict):
            # Check if it's a HuggingFace dataset dict with splits
            if hasattr(self.dataset, 'keys') and 'train' in self.dataset:
                dataset_to_explore = self.dataset['train']
                columns = dataset_to_explore.column_names if hasattr(dataset_to_explore, 'column_names') else list(next(iter(dataset_to_explore.values())).keys())
            else:
                # It's a raw dictionary (our CoT-Collection format)
                dataset_to_explore = self.dataset
                # Get column names from first example
                first_key = next(iter(dataset_to_explore.keys()))
                first_value = dataset_to_explore[first_key]
                if isinstance(first_value, dict):
                    columns = list(first_value.keys())
                else:
                    columns = ['data']  # Fallback column name
        else:
            dataset_to_explore = self.dataset
            columns = dataset_to_explore.column_names if hasattr(dataset_to_explore, 'column_names') else []
        
        # Get sample data
        sample_size = min(3, len(dataset_to_explore))
        
        if isinstance(dataset_to_explore, dict):
            # Raw dictionary format
            sample_keys = list(dataset_to_explore.keys())[:sample_size]
            sample_data = [dataset_to_explore[key] for key in sample_keys]
        else:
            # HuggingFace dataset format
            sample_data = [dataset_to_explore[i] for i in range(sample_size)]
        
        structure_info = {
            'total_samples': len(dataset_to_explore),
            'columns': columns,
            'sample_data': sample_data
        }
        
        logger.info(f"Dataset structure: {structure_info}")
        return structure_info
    
    def process_dataset(self) -> List[Dict[str, str]]:
        """
        Process the dataset to extract (Question, Rationale, Answer) triplets.
        
        Returns:
            List of processed triplets
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Processing dataset to extract triplets...")
        
        processed_triplets = []
        
        # Dataset is {row_id: {field: value, ...}}, iterate directly
        dataset_items = self.dataset.items()
        total_samples = len(self.dataset)
        
        logger.info(f"Processing {total_samples} samples...")
        
        for row_id, row_data in dataset_items:
            try:
                # row_id is the key (string), row_data is the dictionary with fields
                original_id = row_id
                data = row_data
                
                # Extract question, rationale, and answer based on dataset structure
                triplet = self._extract_triplet(data)
                if triplet:
                    triplet['id'] = original_id  # Use original ID from dataset
                    triplet['source'] = 'cot_collection'  # Mark source
                    processed_triplets.append(triplet)
                
                # Log progress
                processed_count = len(processed_triplets)
                if processed_count > 0 and processed_count % 10000 == 0:
                    logger.info(f"Processed {processed_count} valid triplets so far...")
                    
            except Exception as e:
                logger.warning(f"Failed to process sample {row_id}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_triplets)} triplets from {total_samples} total samples")
        self.processed_data = processed_triplets
        return processed_triplets
    
    def _extract_triplet(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract (Question, Rationale, Answer) from a single CoT-Collection example.
        
        Args:
            example: Single example from the CoT-Collection dataset
            
        Returns:
            Dictionary with 'question', 'rationale', 'answer' keys
        """
        triplet = {}
        
        # CoT-Collection structure:
        # - source: The input question/context
        # - target: The final answer
        # - rationale: The chain-of-thought reasoning
        # - task: Task type (e.g., 'math_qa', 'squad_v1', etc.)
        
        try:
            # Extract question from 'source' field
            if 'source' in example and example['source']:
                triplet['question'] = str(example['source']).strip()
            
            # Extract rationale from 'rationale' field
            if 'rationale' in example and example['rationale']:
                triplet['rationale'] = str(example['rationale']).strip()
            
            # Extract answer from 'target' field
            if 'target' in example and example['target']:
                triplet['answer'] = str(example['target']).strip()
            
            # Add task information for context
            if 'task' in example:
                triplet['task'] = str(example['task']).strip()
            
            # Validate that we have the essential fields
            required_fields = ['question', 'rationale', 'answer']
            missing_fields = [field for field in required_fields if field not in triplet or not triplet[field]]
            
            if missing_fields:
                logger.debug(f"Missing required fields {missing_fields} in example: {list(example.keys())}")
                return None
            
            # Filter out very short content (but be more lenient than before)
            if (len(triplet['question']) < 3 or 
                len(triplet['rationale']) < 10 or 
                len(triplet['answer']) < 1):
                logger.debug(f"Content too short: Q={len(triplet['question'])}, R={len(triplet['rationale'])}, A={len(triplet['answer'])}")
                return None
            
            # Clean up the triplet
            for key in triplet:
                # Remove extra whitespace and newlines
                triplet[key] = ' '.join(triplet[key].split())
            
            return triplet
            
        except Exception as e:
            logger.debug(f"Error extracting triplet: {e}")
            return None
    
    def save_processed_data(self, filename: str = "processed_triplets.json") -> Path:
        """
        Save processed triplets to disk.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        if not self.processed_data:
            raise ValueError("No processed data to save. Call process_dataset() first.")
        
        output_path = self.processed_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.processed_data)} triplets to {output_path}")
        return output_path
    
    def load_processed_data(self, filename: str = "processed_triplets.json") -> List[Dict[str, str]]:
        """
        Load previously processed triplets from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            List of processed triplets
        """
        input_path = self.processed_dir / filename
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            self.processed_data = json.load(f)
        
        logger.info(f"Loaded {len(self.processed_data)} triplets from {input_path}")
        return self.processed_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the processed dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.processed_data:
            return {}
        
        stats = {
            'total_triplets': len(self.processed_data),
            'avg_question_length': sum(len(t['question']) for t in self.processed_data) / len(self.processed_data),
            'avg_rationale_length': sum(len(t['rationale']) for t in self.processed_data) / len(self.processed_data),
            'avg_answer_length': sum(len(t['answer']) for t in self.processed_data) / len(self.processed_data),
        }
        
        return stats

def main():
    """Test the dataset processor."""
    import sys
    from pathlib import Path
    
    # Add src to path for config import
    sys.path.append(str(Path(__file__).parents[1]))
    from utils.config_loader import load_config

    # Load configuration
    config = load_config("config/config.yaml")
    
    # Initialize processor
    processor = CoTDatasetProcessor(config)
    
    # Load and explore dataset
    processor.load_dataset()
    structure = processor.explore_dataset_structure()
    
    print("Dataset structure:")
    print(json.dumps(structure, indent=2, default=str))
    
    # Process dataset
    processor.process_dataset()
    
    # Save processed data
    processor.save_processed_data()
    
    # Get statistics
    stats = processor.get_statistics()
    print(f"\nDataset statistics: {stats}")

if __name__ == "__main__":
    main()