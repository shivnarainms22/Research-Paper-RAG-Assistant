"""
FastAPI application for the Research Paper RAG system.
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import get_settings
from src.ingestion import PDFParser, DocumentChunker
from src.retrieval import VectorStore, PaperRetriever
from src.llm import QAEngine


# Request/Response Models
class QuestionRequest(BaseModel):
    """Request model for asking questions."""

    question: str
    paper_filter: Optional[str] = None
    top_k: int = 5


class CompareRequest(BaseModel):
    """Request model for comparing papers."""

    paper_titles: list[str]
    aspect: str


class AnswerResponse(BaseModel):
    """Response model for answers."""

    answer: str
    citations: list[str]
    query_type: str


class PaperInfo(BaseModel):
    """Information about an indexed paper."""

    title: str
    source_file: str
    total_chunks: int


class StatsResponse(BaseModel):
    """Response model for system statistics."""

    total_papers: int
    total_chunks: int
    paper_titles: list[str]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Research Paper RAG System",
        description="AI-powered research paper analysis and question answering",
        version="0.1.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize components
    vector_store = VectorStore()
    retriever = PaperRetriever(vector_store=vector_store)
    qa_engine = QAEngine(retriever=retriever)
    pdf_parser = PDFParser()
    chunker = DocumentChunker()

    # Serve static files
    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def root():
        """Serve the web interface."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {
            "message": "Research Paper RAG System",
            "docs": "/docs",
            "version": "0.1.0",
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/papers/upload")
    async def upload_paper(file: UploadFile = File(...)) -> dict:
        """
        Upload and index a research paper PDF.
        """
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Save the uploaded file
        settings = get_settings()
        papers_dir = settings.papers_dir
        papers_dir.mkdir(parents=True, exist_ok=True)

        file_path = papers_dir / file.filename
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        try:
            # Parse the PDF
            parsed = pdf_parser.parse(file_path)

            # Chunk the document
            chunks = chunker.chunk_paper(parsed)

            # Index in vector store
            num_chunks = vector_store.add_chunks(chunks)

            return {
                "success": True,
                "paper_title": parsed.metadata.title,
                "pages": parsed.metadata.total_pages,
                "chunks_indexed": num_chunks,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    @app.post("/ask", response_model=AnswerResponse)
    async def ask_question(request: QuestionRequest) -> AnswerResponse:
        """
        Ask a question about the indexed papers.
        """
        try:
            answer = qa_engine.ask(
                question=request.question,
                paper_filter=request.paper_filter,
                top_k=request.top_k,
            )

            return AnswerResponse(
                answer=answer.content,
                citations=answer.citations,
                query_type=answer.query_type,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

    @app.post("/compare", response_model=AnswerResponse)
    async def compare_papers(request: CompareRequest) -> AnswerResponse:
        """
        Compare multiple papers on a specific aspect.
        """
        try:
            answer = qa_engine.compare_papers(
                paper_titles=request.paper_titles,
                aspect=request.aspect,
            )

            return AnswerResponse(
                answer=answer.content,
                citations=answer.citations,
                query_type=answer.query_type,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error comparing papers: {str(e)}")

    @app.get("/papers", response_model=list[str])
    async def list_papers() -> list[str]:
        """
        List all indexed papers.
        """
        return qa_engine.get_available_papers()

    @app.delete("/papers/{paper_title}")
    async def delete_paper(paper_title: str) -> dict:
        """
        Delete a paper from the index.
        """
        try:
            deleted_count = vector_store.delete_paper(paper_title)
            return {
                "success": True,
                "paper_title": paper_title,
                "chunks_deleted": deleted_count,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting paper: {str(e)}")

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats() -> StatsResponse:
        """
        Get system statistics.
        """
        stats = vector_store.get_stats()
        return StatsResponse(
            total_papers=stats["total_papers"],
            total_chunks=stats["total_chunks"],
            paper_titles=stats["paper_titles"],
        )

    return app


# Create the app instance
app = create_app()

