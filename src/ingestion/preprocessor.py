"""
Text preprocessing and math extraction utilities.
"""

import re
from dataclasses import dataclass


@dataclass
class ExtractedEquation:
    """Represents an extracted mathematical equation."""

    latex: str
    plain_text: str = ""
    context: str = ""
    equation_type: str = "inline"  # inline, display, numbered


class MathExtractor:
    """
    Extracts and processes mathematical equations from research papers.
    
    Handles LaTeX notation and provides plain-English explanations.
    """

    # Common LaTeX to plain text mappings
    LATEX_TO_TEXT = {
        r"\\alpha": "alpha",
        r"\\beta": "beta",
        r"\\gamma": "gamma",
        r"\\theta": "theta",
        r"\\pi": "pi",
        r"\\sum": "sum of",
        r"\\prod": "product of",
        r"\\int": "integral of",
        r"\\frac\{([^}]+)\}\{([^}]+)\}": r"\1 divided by \2",
        r"\\sqrt\{([^}]+)\}": r"square root of \1",
        r"\^2": " squared",
        r"\^3": " cubed",
        r"\^{([^}]+)}": r" to the power of \1",
        r"_\{([^}]+)\}": r" subscript \1",
        r"\\leq": "≤",
        r"\\geq": "≥",
        r"\\neq": "≠",
        r"\\approx": "≈",
        r"\\infty": "infinity",
        r"\\nabla": "gradient",
        r"\\partial": "partial derivative",
        r"\\mathbb\{E\}": "expected value",
        r"\\mathbb\{R\}": "real numbers",
    }

    def extract_equations(self, text: str) -> list[ExtractedEquation]:
        """
        Extract all equations from text.
        
        Args:
            text: Raw text potentially containing LaTeX equations.
            
        Returns:
            List of ExtractedEquation objects.
        """
        equations = []

        # Display math: $$...$$
        for match in re.finditer(r'\$\$(.+?)\$\$', text, re.DOTALL):
            latex = match.group(1).strip()
            context = self._get_context(text, match.start(), match.end())
            equations.append(ExtractedEquation(
                latex=latex,
                plain_text=self.latex_to_plain(latex),
                context=context,
                equation_type="display",
            ))

        # Inline math: $...$
        for match in re.finditer(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', text):
            latex = match.group(1).strip()
            context = self._get_context(text, match.start(), match.end())
            equations.append(ExtractedEquation(
                latex=latex,
                plain_text=self.latex_to_plain(latex),
                context=context,
                equation_type="inline",
            ))

        # Numbered equations: \begin{equation}...\end{equation}
        for match in re.finditer(r'\\begin\{equation\}(.+?)\\end\{equation\}', text, re.DOTALL):
            latex = match.group(1).strip()
            context = self._get_context(text, match.start(), match.end())
            equations.append(ExtractedEquation(
                latex=latex,
                plain_text=self.latex_to_plain(latex),
                context=context,
                equation_type="numbered",
            ))

        return equations

    def latex_to_plain(self, latex: str) -> str:
        """
        Convert LaTeX notation to plain English.
        
        Args:
            latex: LaTeX equation string.
            
        Returns:
            Plain English representation.
        """
        result = latex

        # Apply conversion rules
        for pattern, replacement in self.LATEX_TO_TEXT.items():
            result = re.sub(pattern, replacement, result)

        # Clean up remaining LaTeX commands
        result = re.sub(r'\\[a-zA-Z]+', '', result)
        result = re.sub(r'[{}]', '', result)
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Get surrounding context for an equation."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()


class TextPreprocessor:
    """
    Preprocesses research paper text for better retrieval and understanding.
    """

    def __init__(self):
        self.math_extractor = MathExtractor()

    def preprocess(self, text: str) -> str:
        """
        Clean and normalize text while preserving important content.
        
        Args:
            text: Raw text from PDF extraction.
            
        Returns:
            Cleaned and normalized text.
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Fix common OCR/extraction issues
        text = self._fix_hyphenation(text)
        text = self._normalize_references(text)

        # Clean up but preserve equations
        text = self._clean_preserve_math(text)

        return text.strip()

    def _fix_hyphenation(self, text: str) -> str:
        """Fix words split across lines with hyphens."""
        # Match word-\nword patterns and join them
        return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    def _normalize_references(self, text: str) -> str:
        """Normalize citation references."""
        # Convert [1,2,3] style to [1, 2, 3]
        def fix_refs(match):
            refs = match.group(1)
            refs = re.sub(r',\s*', ', ', refs)
            return f'[{refs}]'

        return re.sub(r'\[(\d+(?:,\s*\d+)*)\]', fix_refs, text)

    def _clean_preserve_math(self, text: str) -> str:
        """Clean text while preserving mathematical notation."""
        # Preserve content within $ signs
        protected = {}
        counter = [0]

        def protect(match):
            key = f"__MATH_{counter[0]}__"
            protected[key] = match.group(0)
            counter[0] += 1
            return key

        # Protect math content
        text = re.sub(r'\$\$.+?\$\$', protect, text, flags=re.DOTALL)
        text = re.sub(r'\$.+?\$', protect, text)

        # Clean the non-math text
        text = re.sub(r'[^\w\s.,!?;:\'\"\-()[\]{}$\\]', '', text)

        # Restore math content
        for key, value in protected.items():
            text = text.replace(key, value)

        return text

    def extract_key_terms(self, text: str) -> list[str]:
        """Extract key technical terms from text."""
        # Common ML/AI terms to look for
        ml_terms = [
            "neural network", "deep learning", "reinforcement learning",
            "transformer", "attention", "gradient descent", "backpropagation",
            "convolutional", "recurrent", "LSTM", "GRU", "embedding",
            "policy", "value function", "reward", "state", "action",
            "loss function", "optimization", "regularization", "dropout",
            "batch normalization", "activation function", "softmax",
        ]

        found_terms = []
        text_lower = text.lower()

        for term in ml_terms:
            if term.lower() in text_lower:
                found_terms.append(term)

        return found_terms

