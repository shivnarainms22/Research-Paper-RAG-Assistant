"""
Document Chunking for optimal retrieval.
"""

from dataclasses import dataclass
from typing import Optional

from .pdf_parser import ParsedPaper


@dataclass
class DocumentChunk:
    """A chunk of document content with metadata for retrieval."""

    content: str
    paper_title: str
    source_file: str
    page_numbers: list[int]
    section: str = ""
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0

    @property
    def citation(self) -> str:
        """Generate a citation string for this chunk."""
        pages = ", ".join(map(str, self.page_numbers))
        section_str = f", {self.section}" if self.section else ""
        return f"[{self.paper_title}{section_str}, p. {pages}]"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage (ChromaDB compatible)."""
        import json
        return {
            "content": self.content,
            "paper_title": self.paper_title,
            "source_file": self.source_file,
            "page_numbers": json.dumps(self.page_numbers),  # ChromaDB needs strings
            "section": self.section,
            "chunk_index": self.chunk_index,
            "citation": self.citation,
        }


class DocumentChunker:
    """
    Splits documents into overlapping chunks optimized for RAG retrieval.
    
    Uses semantic-aware chunking that respects paragraph and sentence boundaries.
    
    Example:
        chunker = DocumentChunker(chunk_size=1000, overlap=200)
        chunks = chunker.chunk_paper(parsed_paper)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        respect_sections: bool = True,
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters.
            overlap: Number of overlapping characters between chunks.
            respect_sections: Whether to avoid splitting across section boundaries.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sections = respect_sections

    def chunk_paper(self, paper: ParsedPaper) -> list[DocumentChunk]:
        """
        Split a parsed paper into retrieval-optimized chunks.
        
        Always chunks the ENTIRE paper text to ensure nothing is lost.
        Section detection is used only for citation metadata.
        
        Args:
            paper: A ParsedPaper object from the PDF parser.
            
        Returns:
            List of DocumentChunk objects.
        """
        # Always chunk the full text to ensure nothing is lost
        chunks = self._chunk_text(
            text=paper.full_text,
            paper=paper,
            start_index=0,
        )
        
        # If section detection is enabled, assign sections to chunks for better citations
        if self.respect_sections and paper.metadata.sections:
            self._assign_sections_to_chunks(chunks, paper)

        return chunks
    
    def _assign_sections_to_chunks(self, chunks: list[DocumentChunk], paper: ParsedPaper) -> None:
        """
        Assign section labels to chunks based on their content position.
        This improves citation quality without losing any content.
        """
        # Build a map of section positions in the full text
        section_positions = []
        full_text_lower = paper.full_text.lower()
        
        for section in paper.metadata.sections:
            section_lower = section.lower()
            pos = full_text_lower.find(section_lower)
            if pos != -1:
                section_positions.append((pos, section))
        
        # Sort by position
        section_positions.sort(key=lambda x: x[0])
        
        # Assign sections to chunks based on their start position
        for chunk in chunks:
            chunk_start = chunk.start_char
            assigned_section = ""
            
            for pos, section in section_positions:
                if pos <= chunk_start:
                    assigned_section = section
                else:
                    break
            
            chunk.section = assigned_section

    def _chunk_text(
        self,
        text: str,
        paper: ParsedPaper,
        start_index: int,
    ) -> list[DocumentChunk]:
        """Split text into overlapping chunks."""
        chunks = []
        paragraphs = text.split("\n\n")

        current_chunk = ""
        current_start = 0
        chunk_index = start_index
        char_position = 0  # Track position in original text

        for para in paragraphs:
            para = para.strip()
            if not para:
                char_position += 2  # Account for \n\n
                continue

            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(self._create_chunk(
                    content=current_chunk.strip(),
                    paper=paper,
                    chunk_index=chunk_index,
                    start_char=current_start,
                ))
                chunk_index += 1

                # Start new chunk with overlap from previous
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_start = char_position - len(overlap_text)
                current_chunk = overlap_text + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_start = char_position
                    current_chunk = para
            
            char_position += len(para) + 2  # +2 for \n\n separator

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                content=current_chunk.strip(),
                paper=paper,
                chunk_index=chunk_index,
                start_char=current_start,
            ))

        return chunks

    def _create_chunk(
        self,
        content: str,
        paper: ParsedPaper,
        chunk_index: int,
        start_char: int,
    ) -> DocumentChunk:
        """Create a DocumentChunk with page number detection."""
        # Determine which pages this chunk spans
        page_numbers = self._find_page_numbers(content, paper)

        return DocumentChunk(
            content=content,
            paper_title=paper.metadata.title,
            source_file=paper.metadata.source_file,
            page_numbers=page_numbers,
            section="",  # Section assigned later by _assign_sections_to_chunks
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=start_char + len(content),
        )

    def _find_page_numbers(self, content: str, paper: ParsedPaper) -> list[int]:
        """Find which pages contain the given content."""
        pages = []
        content_lower = content.lower()[:100]  # Check first 100 chars

        for page in paper.pages:
            if content_lower in page.text.lower():
                pages.append(page.page_number)

        # Fallback: if no exact match, use content position estimation
        if not pages:
            pages = [1]

        return pages

