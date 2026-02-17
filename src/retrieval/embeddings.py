"""
Embedding generation for document chunks.
"""

from typing import Optional

import numpy as np
from openai import OpenAI

from src.config import get_settings


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using OpenAI's embedding models.
    
    Example:
        generator = EmbeddingGenerator()
        embedding = generator.embed("What is reinforcement learning?")
        embeddings = generator.embed_batch(["text1", "text2", "text3"])
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model: Embedding model name (defaults to config setting).
            api_key: OpenAI API key (defaults to config setting).
        """
        settings = get_settings()
        self.model = model or settings.embedding_model
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        if self._dimension is None:
            # Get dimension by embedding a test string
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            List of floats representing the embedding vector.
        """
        # Clean and truncate text if needed
        text = text.replace("\n", " ").strip()
        if not text:
            text = "empty"

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process in each API call.
            
        Returns:
            List of embedding vectors.
        """
        all_embeddings = []

        # Clean texts
        cleaned_texts = [
            t.replace("\n", " ").strip() or "empty"
            for t in texts
        ]

        # Process in batches
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i:i + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Cosine similarity score between 0 and 1.
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

