"""
High-level retrieval interface for querying research papers.
"""

from dataclasses import dataclass
from typing import Optional

from .vector_store import VectorStore


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""

    content: str
    paper_title: str
    section: str
    page_numbers: list[int]
    citation: str
    similarity: float

    def __str__(self) -> str:
        return f"{self.citation}\n{self.content[:200]}..."


class PaperRetriever:
    """
    High-level interface for retrieving relevant content from research papers.
    
    Supports:
    - Single paper queries
    - Cross-paper comparison
    - Section-specific retrieval
    
    Example:
        retriever = PaperRetriever()
        results = retriever.retrieve("Explain the PPO objective function")
        comparison = retriever.compare_papers(
            ["Paper A", "Paper B"],
            "What are the key differences in their approaches?"
        )
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Optional VectorStore instance.
        """
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        paper_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant content for a query.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            paper_filter: Optional paper title to filter results.
            
        Returns:
            List of RetrievalResult objects.
        """
        raw_results = self.vector_store.search(
            query=query,
            top_k=top_k,
            filter_paper=paper_filter,
        )

        results = []
        for r in raw_results:
            metadata = r.get("metadata", {})

            # Parse page_numbers from metadata (stored as string in ChromaDB)
            page_numbers = metadata.get("page_numbers", [1])
            if isinstance(page_numbers, str):
                try:
                    import json
                    page_numbers = json.loads(page_numbers)
                except:
                    page_numbers = [1]

            results.append(RetrievalResult(
                content=r.get("content", ""),
                paper_title=metadata.get("paper_title", "Unknown"),
                section=metadata.get("section", ""),
                page_numbers=page_numbers,
                citation=r.get("citation", ""),
                similarity=r.get("similarity", 0.0),
            ))

        return results

    def retrieve_for_comparison(
        self,
        paper_titles: list[str],
        query: str,
        results_per_paper: int = 3,
    ) -> dict[str, list[RetrievalResult]]:
        """
        Retrieve relevant content from multiple papers for comparison.
        
        Args:
            paper_titles: List of paper titles to compare.
            query: The comparison query.
            results_per_paper: Number of results per paper.
            
        Returns:
            Dictionary mapping paper titles to their results.
        """
        comparison_results = {}

        for title in paper_titles:
            results = self.retrieve(
                query=query,
                top_k=results_per_paper,
                paper_filter=title,
            )
            comparison_results[title] = results

        return comparison_results

    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 8,
        max_tokens: int = 6000,
        paper_filter: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        """
        Get formatted context and citations for LLM input.
        
        Args:
            query: The user's query.
            top_k: Number of chunks to retrieve.
            max_tokens: Approximate max tokens for context.
            paper_filter: Optional paper title to filter results.
            
        Returns:
            Tuple of (formatted_context, list_of_citations).
        """
        results = self.retrieve(query=query, top_k=top_k, paper_filter=paper_filter)

        context_parts = []
        citations = []
        current_length = 0
        approx_chars_per_token = 4

        for result in results:
            chunk_length = len(result.content) // approx_chars_per_token

            if current_length + chunk_length > max_tokens:
                break

            context_parts.append(
                f"--- From: {result.citation} ---\n{result.content}\n"
            )
            citations.append(result.citation)
            current_length += chunk_length

        context = "\n".join(context_parts)

        return context, citations

    def get_available_papers(self) -> list[str]:
        """Get list of all indexed papers."""
        return self.vector_store.get_paper_titles()

