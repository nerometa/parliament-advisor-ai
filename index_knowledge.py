#!/usr/bin/env python3
"""Index knowledge base for RAG.

This script generates embeddings for all PDF documents.
It requires sentence-transformers (development dependency only).

Usage:
    python index_knowledge.py              # Build index
    python index_knowledge.py --rebuild  # Force rebuild
    python index_knowledge.py --check     # Check if index exists
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("ERROR: sentence-transformers not installed.")
    print("Run: pip install sentence-transformers")
    sys.exit(1)

import config
from rag.chunker import Chunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using pypdf."""
    from pypdf import PdfReader
    
    text_parts = []
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                text_parts.append(text)
    except Exception as e:
        logger.error(f"Failed to extract {pdf_path}: {e}")
    
    return "\n\n".join(text_parts)


def load_knowledge_pdfs(knowledge_dir: Path) -> list:
    """Load all PDFs from knowledge directory."""
    chunks = []
    chunker = Chunker()
    
    pdf_files = list(knowledge_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        
        if text.strip():
            doc_chunks = chunker.chunk(text, source=pdf_path.name)
            for chunk in doc_chunks:
                chunks.append({
                    "text": chunk.text,
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", 0),
                    "article": chunk.metadata.get("article"),
                    "token_count": chunk.metadata.get("token_count", 0),
                })
    
    return chunks


def generate_embeddings(texts: list, model_name: str) -> np.ndarray:
    """Generate embeddings using sentence-transformers."""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info(f"Generating embeddings for {len(texts)} chunks")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return np.array(embeddings)


def save_index(chunks: list, embeddings: np.ndarray, output_dir: Path):
    """Save embeddings and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.json"
    
    logger.info(f"Saving embeddings to {embeddings_path}")
    np.save(embeddings_path, embeddings)
    
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Save stats
    stats = {
        "num_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1],
        "model": config.EMBEDDING_MODEL,
        "size_mb": embeddings_path.stat().st_size / (1024 * 1024),
    }
    
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Index complete: {stats['num_chunks']} chunks, {stats['size_mb']:.1f}MB")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from knowledge PDFs")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if index exists")
    parser.add_argument("--check", action="store_true", help="Check if index exists")
    args = parser.parse_args()
    
    output_dir = Path(config.VECTOR_DB_PATH)
    
    if args.check:
        if (output_dir / "embeddings.npy").exists():
            print(f"✓ Index exists at {output_dir}")
            if (output_dir / "stats.json").exists():
                import json
                with open(output_dir / "stats.json") as f:
                    stats = json.load(f)
                    print(f"  - {stats['num_chunks']} chunks")
                    print(f"  - {stats['size_mb']:.1f}MB embeddings")
            sys.exit(0)
        else:
            print(f"✗ Index not found at {output_dir}")
            sys.exit(1)
    
    if not args.rebuild and (output_dir / "embeddings.npy").exists():
        logger.info(f"Index already exists at {output_dir}")
        logger.info("Use --rebuild to regenerate")
        return
    
    # Build index
    knowledge_dir = Path(config.KNOWLEDGE_DIR)
    if not knowledge_dir.exists():
        logger.error(f"Knowledge directory not found: {knowledge_dir}")
        sys.exit(1)
    
    logger.info("Loading PDFs...")
    chunks = load_knowledge_pdfs(knowledge_dir)
    logger.info(f"Created {len(chunks)} chunks")
    
    if not chunks:
        logger.error("No chunks generated!")
        sys.exit(1)
    
    logger.info("Generating embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts, config.EMBEDDING_MODEL)
    
    logger.info("Saving index...")
    stats = save_index(chunks, embeddings, output_dir)
    
    print(f"\n✓ Index built successfully!")
    print(f"  Location: {output_dir}")
    print(f"  Chunks: {stats['num_chunks']}")
    print(f"  Size: {stats['size_mb']:.1f}MB")
    print(f"  Model: {config.EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
