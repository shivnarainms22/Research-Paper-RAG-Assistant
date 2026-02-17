"""
Retrieval Module
================
Vector store management and semantic search for research papers.
"""

from .vector_store import VectorStore
from .embeddings import EmbeddingGenerator
from .retriever import PaperRetriever

__all__ = ["VectorStore", "EmbeddingGenerator", "PaperRetriever"]

