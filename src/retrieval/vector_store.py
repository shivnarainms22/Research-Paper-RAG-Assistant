"""
Vector store for storing and retrieving document embeddings.
"""

from pathlib import Path
from typing import Optional
import json

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.ingestion.chunker import DocumentChunk
from .embeddings import EmbeddingGenerator


class VectorStore:
    """
    ChromaDB-based vector store for research paper chunks.
    
    Provides persistent storage and semantic search capabilities.
    
    Example:
        store = VectorStore()
        store.add_chunks(chunks)
        results = store.search("What is policy gradient?", top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_dir: Optional[Path] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection.
            persist_dir: Directory for persistent storage.
        """
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Research paper chunks for RAG"},
        )

        self.embedding_generator = EmbeddingGenerator()

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        show_progress: bool = True,
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects.
            show_progress: Whether to show progress bar.
            
        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        # Prepare data
        ids = [f"{chunk.source_file}_{chunk.chunk_index}" for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.to_dict() for chunk in chunks]

        # Generate embeddings
        if show_progress:
            from tqdm import tqdm
            print("Generating embeddings...")

        embeddings = self.embedding_generator.embed_batch(documents)

        # Upsert to collection (handles both new and existing chunks)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_paper: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            filter_paper: Optional paper title to filter results.
            
        Returns:
            List of result dictionaries with content and metadata.
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.embed(query)

        # Build filter if paper specified
        where_filter = None
        if filter_paper:
            where_filter = {"paper_title": filter_paper}

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0

                formatted_results.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "citation": metadata.get("citation", ""),
                })

        return formatted_results

    def get_paper_titles(self) -> list[str]:
        """Get list of all indexed paper titles."""
        all_metadata = self.collection.get(include=["metadatas"])
        titles = set()
        for metadata in all_metadata.get("metadatas", []):
            if metadata and "paper_title" in metadata:
                titles.add(metadata["paper_title"])
        return sorted(titles)

    def delete_paper(self, paper_title: str) -> int:
        """
        Delete all chunks for a specific paper.
        
        Args:
            paper_title: Title of the paper to delete.
            
        Returns:
            Number of chunks deleted.
        """
        # Get IDs for this paper
        results = self.collection.get(
            where={"paper_title": paper_title},
            include=["metadatas"],
        )

        ids_to_delete = results.get("ids", [])

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        count = self.collection.count()
        titles = self.get_paper_titles()

        return {
            "total_chunks": count,
            "total_papers": len(titles),
            "paper_titles": titles,
        }

