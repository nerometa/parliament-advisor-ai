# Minimal RAG Setup Guide

This project now uses a **minimal-dependency RAG** (Retrieval-Augmented Generation) system that requires only **numpy** at runtime.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DEVELOPMENT (one-time setup)                                │
│  ├─ Install sentence-transformers (~100MB)                  │
│  ├─ Run: python index_knowledge.py                          │
│  └─ Generates: embeddings.npy + metadata.json (~5MB)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  RUNTIME (production)                                        │
│  ├─ Dependencies: numpy only (~10MB total)                │
│  ├─ Loads: Pre-computed embeddings                        │
│  └─ Search: Cosine similarity via numpy                   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Initial Setup (Development)

Install development dependencies to build the index:

```bash
pip install sentence-transformers pypdf
python index_knowledge.py
```

This creates:
- `data/vector_store/embeddings.npy` - Document embeddings
- `data/vector_store/metadata.json` - Document metadata
- `data/vector_store/stats.json` - Index statistics

### 2. Runtime (Production)

Install only runtime dependencies:

```bash
pip install -r requirements.txt  # Installs only numpy + core deps
python main.py
```

### 3. Adding New PDFs

When you add new PDFs to `knowledge/`:

```bash
python index_knowledge.py --rebuild
```

## Commands

### `index_knowledge.py`

```bash
# Build index
python index_knowledge.py

# Force rebuild
python index_knowledge.py --rebuild

# Check if index exists
python index_knowledge.py --check
```

### Testing RAG

```python
from rag.minimal_rag import ThaiRAG
import numpy as np

# Load index
rag = ThaiRAG()

# Check status
print(rag.get_stats())

# Search (requires query embedding)
query_emb = np.random.randn(768)  # Replace with actual embedding
results = rag.search(query_emb, k=5)
for r in results:
    print(f"Score: {r['score']:.3f}")
    print(f"Text: {r['text'][:100]}...")
    print(f"Source: {r['metadata']['source']}")
```

## Configuration

Edit `.env` or set environment variables:

```bash
VECTOR_DB_PATH=data/vector_store        # Path to store embeddings
EMBEDDING_MODEL=KoonJamesZ/sentence-transformers-nina-thai-v3
CHUNK_SIZE_TOKENS=512                   # Tokens per chunk
CHUNK_OVERLAP_TOKENS=50                 # Overlap between chunks
TOP_K_CHUNKS=5                          # Number of results to retrieve
```

## File Structure

```
.
├── config.py                    # Configuration (reads .env)
├── core.py                      # Gemini Live session + RAG integration
├── index_knowledge.py           # Build embeddings (dev tool)
├── requirements.txt             # Minimal dependencies
├── rag/
│   ├── __init__.py
│   ├── chunker.py              # Text chunking
│   └── minimal_rag.py          # Zero-dep RAG (numpy only)
├── data/
│   └── vector_store/           # Generated embeddings
│       ├── embeddings.npy
│       ├── metadata.json
│       └── stats.json
└── knowledge/                  # Your PDFs
    ├── รัฐธรรมนูญ 60.PDF
    └── ...
```

## Troubleshooting

### "RAG not initialized" error

Run the indexer:
```bash
python index_knowledge.py
```

### ImportError for sentence-transformers

This is expected at runtime. Only needed during indexing:
```bash
# For indexing only:
pip install sentence-transformers

# For runtime:
pip install -r requirements.txt
```

### Quota exceeded error still happens

Check that core.py no longer loads knowledge into system_instruction:
```python
# Old (loads 340KB):
system_text = "\n\n---\n\n".join([system_text, knowledge_text])

# New (loads only system_prompt.txt):
# RAG is used via retrieve_context() method
```

## Migration from Old Stack

If you had the old heavy stack installed:

```bash
# Remove old deps
pip uninstall -y sentence-transformers lancedb torch transformers scikit-learn scipy

# Install minimal deps
pip install -r requirements.txt

# Rebuild index
python index_knowledge.py
```

## Performance

- **Index Size**: ~5MB for 9 PDFs (~340KB text)
- **Search Latency**: <10ms for 1000 chunks
- **Memory Usage**: ~10MB (embeddings + code)
- **Cold Start**: Instant (no model loading)

## Architecture Decisions

1. **Pre-computed Embeddings**: Build once, use many times
2. **Numpy Only**: Zero ML dependencies at runtime
3. **Cosine Similarity**: Fast enough for <1000 documents
4. **JSON Metadata**: Human-readable, easy to debug

## Future Enhancements

- Add API-based embeddings (OpenAI) for zero local ML
- Incremental indexing (add new PDFs without rebuild)
- Hybrid search (keywords + embeddings)
- LRU cache for frequent queries

## License

Same as parent project.
