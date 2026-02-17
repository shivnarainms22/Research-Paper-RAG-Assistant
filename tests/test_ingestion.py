"""
Tests for the ingestion module.
"""

import pytest
from src.ingestion.preprocessor import MathExtractor, TextPreprocessor
from src.ingestion.chunker import DocumentChunk


class TestMathExtractor:
    """Tests for MathExtractor."""

    def test_latex_to_plain_basic(self):
        """Test basic LaTeX to plain text conversion."""
        extractor = MathExtractor()
        
        # Test Greek letters
        assert "alpha" in extractor.latex_to_plain(r"\alpha")
        assert "beta" in extractor.latex_to_plain(r"\beta")
        
    def test_latex_to_plain_fractions(self):
        """Test fraction conversion."""
        extractor = MathExtractor()
        result = extractor.latex_to_plain(r"\frac{a}{b}")
        assert "divided by" in result

    def test_extract_equations_inline(self):
        """Test inline equation extraction."""
        extractor = MathExtractor()
        text = "The equation $E = mc^2$ is famous."
        equations = extractor.extract_equations(text)
        
        assert len(equations) >= 1
        assert any("E = mc^2" in eq.latex for eq in equations)


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def test_fix_hyphenation(self):
        """Test hyphenation fixing."""
        preprocessor = TextPreprocessor()
        text = "This is a hyph-\nenated word."
        result = preprocessor.preprocess(text)
        
        assert "hyphenated" in result

    def test_extract_key_terms(self):
        """Test key term extraction."""
        preprocessor = TextPreprocessor()
        text = "We use deep learning and reinforcement learning with a transformer architecture."
        terms = preprocessor.extract_key_terms(text)
        
        assert "deep learning" in terms
        assert "reinforcement learning" in terms
        assert "transformer" in terms


class TestDocumentChunk:
    """Tests for DocumentChunk."""

    def test_citation_format(self):
        """Test citation string formatting."""
        chunk = DocumentChunk(
            content="Test content",
            paper_title="Attention Is All You Need",
            source_file="attention.pdf",
            page_numbers=[3, 4],
            section="Methods",
            chunk_index=0,
        )
        
        citation = chunk.citation
        assert "Attention Is All You Need" in citation
        assert "Methods" in citation
        assert "3" in citation

    def test_to_dict(self):
        """Test dictionary conversion."""
        chunk = DocumentChunk(
            content="Test content",
            paper_title="Test Paper",
            source_file="test.pdf",
            page_numbers=[1],
            section="Intro",
            chunk_index=0,
        )
        
        d = chunk.to_dict()
        assert d["content"] == "Test content"
        assert d["paper_title"] == "Test Paper"
        assert "citation" in d

