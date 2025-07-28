# CoT Optimization Project

A Chain of Thought (CoT) optimization system for language models using retrieval-augmented generation (RAG) with the CoT-Collection dataset.

## 🎯 Project Overview

This project implements a three-phase pipeline to enhance language model reasoning through retrieval-augmented Chain of Thought prompting:

1. **Phase 1**: Build knowledge base from CoT-Collection dataset with embeddings and FAISS indexing
2. **Phase 2**: Implement retrieval pipeline for finding similar reasoning examples  
3. **Phase 3**: Generate responses using baseline vs CoT comparison methodology

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Phase 1       │    │   Phase 2       │    │   Phase 3       │
│ Knowledge Base  │───▶│   Retrieval     │───▶│   Generation    │
│                 │    │                 │    │                 │
│ • Dataset       │    │ • Query Encode  │    │ • Baseline      │
│ • Embeddings    │    │ • FAISS Search  │    │ • CoT w/ RAG    │
│ • FAISS Index   │    │ • Top-5 Examples│    │ • Comparison    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📂 Project Structure

```
COTProject/
├── src/
│   ├── phase1/              # Knowledge base creation
│   │   ├── dataset_processor.py    # CoT-Collection data processing
│   │   ├── embedding_generator.py  # Question embeddings (all-mpnet-base-v2)
│   │   └── vector_indexer.py       # FAISS index creation/management
│   ├── phase2/              # Retrieval pipeline
│   │   └── retrieval_pipeline.py   # Query processing & example retrieval
│   ├── phase3/              # Generation pipeline
│   │   └── generation_pipeline.py  # T5 baseline vs CoT comparison
│   └── utils/
│       └── config_loader.py        # Configuration management
├── config/
│   └── config.yaml          # System configuration
├── tests/                   # Test files and debugging scripts
├── data/                    # Local data (git-ignored)
│   ├── processed/           # Processed datasets and embeddings
│   ├── faiss_index/         # FAISS vector indices
│   └── cache/               # HuggingFace cache
├── main.py                  # Main pipeline execution
├── demo.py                  # Interactive demonstration
└── requirements.txt         # Python dependencies
```

## 🚀 Key Features

### Phase 1: Knowledge Base Creation
- **Dataset Processing**: Loads CoT-Collection (1.8M examples), extracts question-rationale-answer triplets
- **Smart Embeddings**: Uses `all-mpnet-base-v2` to encode questions for semantic similarity
- **FAISS Indexing**: Creates efficient vector index for fast similarity search
- **Configurable Sampling**: Supports limiting dataset size for development/testing

### Phase 2: Retrieval Pipeline  
- **Semantic Search**: Finds top-5 most similar questions to user queries
- **Multi-format Output**: Simple, detailed, and prompt-ready formatting
- **Similarity Filtering**: Configurable similarity thresholds
- **CoT Prompt Generation**: Creates reasoning examples without answer leakage

### Phase 3: Generation Pipeline
- **Dual Methodology**: Baseline (direct) vs CoT (reasoning-guided) approaches
- **T5 Integration**: Uses T5 model for text generation
- **Retrieval-Augmented CoT**: Incorporates similar reasoning patterns
- **Performance Comparison**: Evaluates baseline vs enhanced reasoning

## 💡 Chain of Thought Strategy

The system implements retrieval-augmented CoT reasoning:

1. **Query Analysis**: User asks "What is 25 × 17?"
2. **Similar Example Retrieval**: Finds 5 similar math problems with step-by-step reasoning
3. **Prompt Construction**: Shows reasoning patterns without revealing answers
4. **Guided Generation**: Model follows demonstrated thinking patterns to solve new problem

### Example Prompt Structure:
```
Here are examples demonstrating how to think step-by-step:

Example 1:
Question: What is 12 × 8?
Rationale: I need to multiply 12 by 8. I can break this down: 12 × 8 = 12 × (10 - 2) = (12 × 10) - (12 × 2) = 120 - 24 = 96.

[... 4 more examples ...]

---

Now, solve the following problem using the same step-by-step thinking:
What is 25 × 17?
```

## 🛠️ Technical Implementation

### Core Technologies
- **HuggingFace Transformers**: T5 model, sentence-transformers
- **FAISS**: Fast similarity search and clustering  
- **CoT-Collection Dataset**: 1.8M question-rationale-answer pairs
- **Sentence Transformers**: all-mpnet-base-v2 for embeddings

### Data Processing Pipeline
```python
# Simplified data flow
raw_data → triplet_extraction → question_embedding → faiss_indexing
query → query_embedding → similarity_search → prompt_construction → generation
```

### Key Optimizations
- **Batch Processing**: Efficient embedding generation
- **Memory Management**: Configurable dataset sampling
- **Index Persistence**: Save/load FAISS indices
- **Path Resolution**: Robust configuration loading

## 🏃‍♂️ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Full Pipeline**:
   ```bash
   python main.py
   ```

3. **Interactive Demo**:
   ```bash
   python demo.py
   ```

4. **Test Individual Components**:
   ```bash
   python tests/test_5k_dataset.py
   python src/phase2/retrieval_pipeline.py
   ```

## 📊 Results & Evaluation

The system enables comparison between:
- **Baseline**: Direct question → answer generation
- **CoT-RAG**: Question → similar examples → reasoning-guided generation

### Expected Improvements
- Enhanced reasoning quality through example-guided thinking
- Better step-by-step problem decomposition  
- Improved accuracy on complex reasoning tasks
- Consistent reasoning patterns across similar problem types

## 🔧 Configuration

Key settings in `config/config.yaml`:
- **Dataset**: Sample size, processing options
- **Embeddings**: Model selection, batch size, device
- **Vector Index**: FAISS index type, similarity thresholds
- **Generation**: Model parameters, output formatting

## 🤝 Development Notes

### Data Management
- Heavy data files (embeddings, indices) are git-ignored
- Data regenerated via pipeline execution
- Configurable sampling for development efficiency

### Testing Strategy
- Component-level tests for each phase
- Integration tests for full pipeline
- Sample data generation for rapid iteration
- Debug utilities for troubleshooting

### Performance Considerations
- CUDA support for GPU acceleration
- Batch processing for efficiency
- Configurable memory usage
- Index persistence for faster startup

## 📝 Research Context

This implementation is based on Chain of Thought prompting research, specifically focusing on:
- Retrieval-augmented reasoning enhancement
- Example-guided problem solving
- Semantic similarity for reasoning pattern matching
- Large-scale reasoning dataset utilization (CoT-Collection)

## 🚀 Future Enhancements

- Multi-domain reasoning support
- Dynamic similarity threshold adjustment
- Advanced prompt engineering techniques
- Performance benchmarking against standard datasets
- Support for additional language models

---

**Built with ❤️ for advancing AI reasoning capabilities**