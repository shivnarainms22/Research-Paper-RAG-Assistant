"""
PDF Parser for extracting text, equations, and metadata from research papers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


@dataclass
class PageContent:
    """Represents content extracted from a single PDF page."""

    page_number: int
    text: str
    equations: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


@dataclass
class PaperMetadata:
    """Metadata extracted from a research paper."""

    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    source_file: str = ""
    total_pages: int = 0
    sections: list[str] = field(default_factory=list)


@dataclass
class ParsedPaper:
    """Complete parsed representation of a research paper."""

    metadata: PaperMetadata
    pages: list[PageContent]
    full_text: str = ""

    def get_page_text(self, page_num: int) -> str:
        """Get text content for a specific page."""
        for page in self.pages:
            if page.page_number == page_num:
                return page.text
        return ""

    def get_section(self, section_name: str) -> str:
        """Extract text for a named section (e.g., 'Methods', 'Results')."""
        # Simple section extraction - can be enhanced with better parsing
        section_lower = section_name.lower()
        lines = self.full_text.split("\n")
        in_section = False
        section_text = []

        for line in lines:
            line_lower = line.lower().strip()
            # Check if we've hit the target section
            if section_lower in line_lower and len(line.strip()) < 50:
                in_section = True
                continue
            # Check if we've hit the next section
            if in_section and self._is_section_header(line):
                break
            if in_section:
                section_text.append(line)

        return "\n".join(section_text)

    def _is_section_header(self, line: str) -> bool:
        """Check if a line looks like a section header."""
        common_sections = [
            "abstract", "introduction", "background", "related work",
            "methods", "methodology", "approach", "experiments",
            "results", "discussion", "conclusion", "references",
            "appendix", "acknowledgments"
        ]
        line_lower = line.lower().strip()
        return any(section in line_lower for section in common_sections) and len(line.strip()) < 50


class PDFParser:
    """
    Parses PDF research papers extracting text, equations, and structure.
    
    Example:
        parser = PDFParser()
        paper = parser.parse("path/to/paper.pdf")
        print(paper.metadata.title)
        print(paper.get_section("Methods"))
    """

    def __init__(self, extract_images: bool = False):
        """
        Initialize the PDF parser.
        
        Args:
            extract_images: Whether to extract and process images/figures.
        """
        self.extract_images = extract_images

    def parse(self, pdf_path: str | Path) -> ParsedPaper:
        """
        Parse a PDF file and extract all content.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            ParsedPaper object with all extracted content.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        pages = []
        full_text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text_parts.append(text)

            # Extract equations (LaTeX patterns)
            equations = self._extract_equations(text)

            pages.append(PageContent(
                page_number=page_num + 1,  # 1-indexed for human readability
                text=text,
                equations=equations,
            ))

        full_text = "\n\n".join(full_text_parts)
        metadata = self._extract_metadata(doc, pdf_path, full_text)

        doc.close()

        return ParsedPaper(
            metadata=metadata,
            pages=pages,
            full_text=full_text,
        )

    def _extract_metadata(
        self, doc: fitz.Document, pdf_path: Path, full_text: str
    ) -> PaperMetadata:
        """Extract metadata from the PDF document."""
        pdf_metadata = doc.metadata

        # Try to get title from PDF metadata, fallback to first line
        title = pdf_metadata.get("title", "")
        if not title:
            first_lines = full_text.split("\n")[:5]
            title = max(first_lines, key=len) if first_lines else pdf_path.stem

        # Extract authors (this is a simplified version)
        authors = []
        if pdf_metadata.get("author"):
            authors = [a.strip() for a in pdf_metadata["author"].split(",")]

        # Extract abstract
        abstract = self._extract_abstract(full_text)

        return PaperMetadata(
            title=title.strip(),
            authors=authors,
            abstract=abstract,
            source_file=str(pdf_path),
            total_pages=len(doc),
            sections=self._detect_sections(full_text),
        )

    def _extract_abstract(self, text: str) -> str:
        """Extract the abstract from the paper text."""
        text_lower = text.lower()
        abstract_start = text_lower.find("abstract")

        if abstract_start == -1:
            return ""

        # Find the end of abstract (usually "introduction" or "1.")
        abstract_text = text[abstract_start + 8:]  # Skip "abstract"
        end_markers = ["introduction", "1.", "1 "]

        end_pos = len(abstract_text)
        for marker in end_markers:
            pos = abstract_text.lower().find(marker)
            if pos != -1 and pos < end_pos:
                end_pos = pos

        return abstract_text[:end_pos].strip()

    def _extract_equations(self, text: str) -> list[str]:
        """Extract LaTeX-style equations from text."""
        import re
        equations = []

        # Match inline math: $...$
        inline = re.findall(r'\$([^$]+)\$', text)
        equations.extend(inline)

        # Match display math: $$...$$
        display = re.findall(r'\$\$([^$]+)\$\$', text)
        equations.extend(display)

        # Match common equation patterns
        eq_patterns = re.findall(r'\\begin\{equation\}(.+?)\\end\{equation\}', text, re.DOTALL)
        equations.extend(eq_patterns)

        return equations

    def _detect_sections(self, text: str) -> list[str]:
        """Detect major sections in the paper."""
        common_sections = [
            "Abstract", "Introduction", "Background", "Related Work",
            "Methods", "Methodology", "Approach", "Experiments",
            "Results", "Discussion", "Conclusion", "References"
        ]

        found_sections = []
        text_lower = text.lower()

        for section in common_sections:
            if section.lower() in text_lower:
                found_sections.append(section)

        return found_sections

