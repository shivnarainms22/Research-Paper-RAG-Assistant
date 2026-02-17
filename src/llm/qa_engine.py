"""
Question-Answering Engine combining retrieval and LLM.
"""

from dataclasses import dataclass
from typing import Generator, Optional

from .chat import ChatLLM
from .prompts import PromptTemplates
from src.retrieval import PaperRetriever


@dataclass
class Answer:
    """Complete answer with sources."""

    content: str
    citations: list[str]
    query_type: str
    tokens_used: int


class QAEngine:
    """
    Main question-answering engine for the Research Paper RAG system.
    
    Combines semantic retrieval with LLM generation to answer questions
    about research papers with accurate citations.
    
    Example:
        engine = QAEngine()
        
        # Simple question
        answer = engine.ask("Explain PPO like I'm new to RL")
        print(answer.content)
        print("Sources:", answer.citations)
        
        # Compare papers
        answer = engine.ask("How is DQN different from PPO?")
        
        # Stream response
        for chunk in engine.ask_stream("What are the limitations?"):
            print(chunk, end="")
    """

    def __init__(
        self,
        retriever: Optional[PaperRetriever] = None,
        llm: Optional[ChatLLM] = None,
    ):
        """
        Initialize the QA engine.
        
        Args:
            retriever: Optional PaperRetriever instance.
            llm: Optional ChatLLM instance.
        """
        self.retriever = retriever or PaperRetriever()
        self.llm = llm or ChatLLM()
        self.prompts = PromptTemplates()

    def ask(
        self,
        question: str,
        paper_filter: Optional[str] = None,
        top_k: int = 5,
        query_type: Optional[str] = None,
    ) -> Answer:
        """
        Ask a question and get an answer with citations.
        
        Args:
            question: The user's question.
            paper_filter: Optional paper title to focus on.
            top_k: Number of chunks to retrieve.
            query_type: Force a specific query type.
            
        Returns:
            Answer object with content and citations.
        """
        # Detect query type if not specified
        if query_type is None:
            query_type = self.prompts.detect_query_type(question)

        # Retrieve relevant context
        context, citations = self.retriever.get_context_for_llm(
            query=question,
            top_k=top_k,
            paper_filter=paper_filter,
        )

        if not context:
            return Answer(
                content="I couldn't find relevant information in the indexed papers. Please make sure papers are ingested first.",
                citations=[],
                query_type=query_type,
                tokens_used=0,
            )

        # Build prompt
        template = self.prompts.get_template(query_type)
        prompt = template.format(context=context, question=question)

        # Get LLM response
        response = self.llm.chat(
            message=prompt,
            system_prompt=self.prompts.SYSTEM_PROMPT,
        )

        return Answer(
            content=response.content,
            citations=citations,
            query_type=query_type,
            tokens_used=response.tokens_used,
        )

    def ask_stream(
        self,
        question: str,
        paper_filter: Optional[str] = None,
        top_k: int = 5,
        query_type: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Ask a question and stream the response.
        
        Args:
            question: The user's question.
            paper_filter: Optional paper title to focus on.
            top_k: Number of chunks to retrieve.
            query_type: Force a specific query type.
            
        Yields:
            Response text chunks.
        """
        # Detect query type if not specified
        if query_type is None:
            query_type = self.prompts.detect_query_type(question)

        # Retrieve relevant context
        context, citations = self.retriever.get_context_for_llm(
            query=question,
            top_k=top_k,
            paper_filter=paper_filter,
        )

        if not context:
            yield "I couldn't find relevant information in the indexed papers. Please make sure papers are ingested first."
            return

        # Build prompt
        template = self.prompts.get_template(query_type)
        prompt = template.format(context=context, question=question)

        # Stream LLM response
        yield from self.llm.stream(
            message=prompt,
            system_prompt=self.prompts.SYSTEM_PROMPT,
        )

        # Yield citations at the end
        if citations:
            yield "\n\n---\n**Sources:**\n"
            for citation in citations:
                yield f"- {citation}\n"

    def compare_papers(
        self,
        paper_titles: list[str],
        aspect: str,
    ) -> Answer:
        """
        Compare multiple papers on a specific aspect.
        
        Args:
            paper_titles: List of paper titles to compare.
            aspect: The aspect to compare (e.g., "methodology", "results").
            
        Returns:
            Answer with comparison analysis.
        """
        # Get results from each paper
        comparison_results = self.retriever.retrieve_for_comparison(
            paper_titles=paper_titles,
            query=aspect,
            results_per_paper=3,
        )

        # Build context from all papers
        context_parts = []
        citations = []

        for title, results in comparison_results.items():
            context_parts.append(f"\n### {title}\n")
            for result in results:
                context_parts.append(f"{result.content}\n{result.citation}\n")
                citations.append(result.citation)

        context = "\n".join(context_parts)

        # Build comparison question
        question = f"Compare these papers regarding: {aspect}"

        # Build prompt
        template = self.prompts.get_template("compare")
        prompt = template.format(context=context, question=question)

        # Get response
        response = self.llm.chat(
            message=prompt,
            system_prompt=self.prompts.SYSTEM_PROMPT,
        )

        return Answer(
            content=response.content,
            citations=citations,
            query_type="compare",
            tokens_used=response.tokens_used,
        )

    def explain_simply(self, concept: str, expertise_level: str = "beginner") -> Answer:
        """
        Explain a concept at a specific expertise level.
        
        Args:
            concept: The concept to explain.
            expertise_level: "beginner", "intermediate", or "expert".
            
        Returns:
            Answer with explanation.
        """
        level_prompts = {
            "beginner": "Explain this like I'm completely new to the field. Use simple analogies and avoid jargon.",
            "intermediate": "Explain this assuming I understand basic ML concepts but am new to this specific topic.",
            "expert": "Provide a detailed technical explanation with mathematical foundations.",
        }

        question = f"{concept}\n\n{level_prompts.get(expertise_level, level_prompts['beginner'])}"

        return self.ask(question, query_type="qa")

    def get_available_papers(self) -> list[str]:
        """Get list of indexed papers."""
        return self.retriever.get_available_papers()

