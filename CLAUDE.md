# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chain of Thought (CoT) embeddings optimization project that implements a retrieval-augmented generation system for improving reasoning in language models. The system builds a searchable knowledge base of reasoning examples and uses them to guide model generation through in-context learning.

Based on the paper: https://aclanthology.org/2023.emnlp-main.782.pdf
Dataset: https://huggingface.co/datasets/kaist-ai/CoT-Collection

## Development Environment

- **IDE**: PyCharm (configuration files in `.idea/`)
- **Python Environment**: Virtual environment located in `.venv/`
- **Language**: Python
- **Key Dependencies**: transformers, sentence-transformers, faiss-cpu, datasets

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Development Commands
```bash
# Run the full pipeline
python main.py

# Run individual phases
python -m src.phase1.build_knowledge_base  # Build vector index
python -m src.phase2.retrieval_pipeline    # Test retrieval
python -m src.phase3.generation_pipeline   # Test generation

# Process dataset and build index
python -m src.phase1.dataset_processor
python -m src.phase1.embedding_generator
python -m src.phase1.vector_indexer
```

## Project Structure

```
├── src/
│   ├── phase1/          # Build and Index the Rationale Knowledge Base
│   │   ├── dataset_processor.py      # Load CoT-Collection dataset
│   │   ├── embedding_generator.py    # Generate question embeddings
│   │   └── vector_indexer.py         # Create FAISS index
│   ├── phase2/          # The Retrieval Pipeline
│   │   └── retrieval_pipeline.py     # Semantic search and retrieval
│   ├── phase3/          # The Generation Pipeline
│   │   └── generation_pipeline.py    # CoT-T5 generation with examples
│   └── utils/           # Shared utilities
│       └── config_loader.py          # Configuration management
├── data/
│   ├── cache/           # Cached datasets
│   ├── processed/       # Processed data
│   └── faiss_index/     # Vector index files
├── config/
│   └── config.yaml      # Project configuration
├── requirements.txt     # Python dependencies
└── main.py             # Main entry point
```

## System Architecture

The project implements a three-phase pipeline:

1. **Phase 1**: Creates a vector database of 1.84M question-rationale pairs from CoT-Collection
2. **Phase 2**: Retrieves semantically similar reasoning examples for new questions
3. **Phase 3**: Uses retrieved examples as few-shot prompts for CoT-T5 model generation

## Configuration

Project settings are managed in `config/config.yaml`:
- Dataset and model configurations
- Vector index parameters
- Retrieval and generation settings
- System preferences (device, paths, etc.)

## Key Models Used

- **Embedding Model**: all-mpnet-base-v2 (sentence-transformers)
- **Generation Model**: google/flan-t5-large (as CoT-T5 alternative)
- **Vector Search**: FAISS IndexFlatIP for cosine similarity