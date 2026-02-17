"""
Tests for the prompt templates.
"""

import pytest
from src.llm.prompts import PromptTemplates


class TestPromptTemplates:
    """Tests for PromptTemplates."""

    def test_detect_query_type_math(self):
        """Test math query detection."""
        assert PromptTemplates.detect_query_type("Explain this equation") == "math"
        assert PromptTemplates.detect_query_type("What does this formula mean?") == "math"

    def test_detect_query_type_compare(self):
        """Test comparison query detection."""
        assert PromptTemplates.detect_query_type("How is DQN different from PPO?") == "compare"
        assert PromptTemplates.detect_query_type("Compare these methods") == "compare"

    def test_detect_query_type_summarize(self):
        """Test summarization query detection."""
        assert PromptTemplates.detect_query_type("Summarize the methods section") == "summarize"
        assert PromptTemplates.detect_query_type("Give me an overview") == "summarize"

    def test_detect_query_type_limitations(self):
        """Test limitations query detection."""
        assert PromptTemplates.detect_query_type("What are the limitations?") == "limitations"
        assert PromptTemplates.detect_query_type("What problems does this have?") == "limitations"

    def test_detect_query_type_default(self):
        """Test default query type."""
        assert PromptTemplates.detect_query_type("Explain PPO") == "qa"
        assert PromptTemplates.detect_query_type("What is attention?") == "qa"

    def test_get_template(self):
        """Test template retrieval."""
        qa_template = PromptTemplates.get_template("qa")
        assert "{context}" in qa_template
        assert "{question}" in qa_template

        math_template = PromptTemplates.get_template("math")
        assert "equation" in math_template.lower()

    def test_system_prompt_exists(self):
        """Test that system prompt exists and is meaningful."""
        assert len(PromptTemplates.SYSTEM_PROMPT) > 100
        assert "research" in PromptTemplates.SYSTEM_PROMPT.lower()

