"""
Command-line interface for the Research Paper RAG system.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config import get_settings
from src.ingestion import PDFParser, DocumentChunker
from src.retrieval import VectorStore, PaperRetriever
from src.llm import QAEngine

app = typer.Typer(
    name="paper-rag",
    help="Research Paper RAG System - Ask questions about ML/AI research papers",
    add_completion=False,
)
console = Console()


def get_engine() -> QAEngine:
    """Initialize and return the QA engine."""
    return QAEngine()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to PDF file or directory of PDFs"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search directories"),
):
    """
    Ingest research papers from PDF files.
    
    Examples:
        paper-rag ingest paper.pdf
        paper-rag ingest ./papers/ --recursive
    """
    parser = PDFParser()
    chunker = DocumentChunker()
    vector_store = VectorStore()

    # Collect PDF files
    if path.is_file():
        pdf_files = [path] if path.suffix.lower() == ".pdf" else []
    else:
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(path.glob(pattern))

    if not pdf_files:
        console.print("[red]No PDF files found![/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(pdf_files)} PDF file(s) to process[/cyan]\n")

    total_chunks = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for pdf_path in pdf_files:
            task = progress.add_task(f"Processing {pdf_path.name}...", total=None)

            try:
                # Parse PDF
                parsed = parser.parse(pdf_path)
                progress.update(task, description=f"Chunking {pdf_path.name}...")

                # Chunk document
                chunks = chunker.chunk_paper(parsed)
                progress.update(task, description=f"Indexing {pdf_path.name}...")

                # Index chunks
                num_chunks = vector_store.add_chunks(chunks, show_progress=False)
                total_chunks += num_chunks

                progress.update(task, description=f"[green]✓ {pdf_path.name} ({num_chunks} chunks)[/green]")

            except Exception as e:
                progress.update(task, description=f"[red]✗ {pdf_path.name}: {str(e)}[/red]")

    console.print(f"\n[green]Successfully indexed {total_chunks} chunks from {len(pdf_files)} paper(s)[/green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the papers"),
    paper: Optional[str] = typer.Option(None, "--paper", "-p", help="Focus on a specific paper"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream the response"),
):
    """
    Ask a question about indexed research papers.
    
    Examples:
        paper-rag ask "Explain PPO like I'm new to RL"
        paper-rag ask "What is the main contribution?" --paper "Attention Is All You Need"
    """
    engine = get_engine()

    console.print(Panel(question, title="Question", border_style="cyan"))
    console.print()

    if stream:
        console.print("[bold]Answer:[/bold]\n")
        for chunk in engine.ask_stream(question, paper_filter=paper):
            console.print(chunk, end="")
        console.print()
    else:
        with console.status("[bold cyan]Thinking...[/bold cyan]"):
            answer = engine.ask(question, paper_filter=paper)

        console.print(Panel(Markdown(answer.content), title="Answer", border_style="green"))

        if answer.citations:
            console.print("\n[bold]Sources:[/bold]")
            for citation in answer.citations:
                console.print(f"  • {citation}")


@app.command()
def compare(
    papers: list[str] = typer.Argument(..., help="Paper titles to compare"),
    aspect: str = typer.Option("methodology", "--aspect", "-a", help="Aspect to compare"),
):
    """
    Compare multiple papers on a specific aspect.
    
    Examples:
        paper-rag compare "Paper A" "Paper B" --aspect "training methodology"
    """
    engine = get_engine()

    console.print(f"[cyan]Comparing papers on: {aspect}[/cyan]\n")

    with console.status("[bold cyan]Analyzing papers...[/bold cyan]"):
        answer = engine.compare_papers(papers, aspect)

    console.print(Panel(Markdown(answer.content), title="Comparison", border_style="green"))


@app.command(name="list")
def list_papers():
    """
    List all indexed papers.
    """
    vector_store = VectorStore()
    stats = vector_store.get_stats()

    if not stats["paper_titles"]:
        console.print("[yellow]No papers indexed yet. Use 'paper-rag ingest' to add papers.[/yellow]")
        return

    table = Table(title="Indexed Papers")
    table.add_column("Paper Title", style="cyan")

    for title in stats["paper_titles"]:
        table.add_row(title)

    console.print(table)
    console.print(f"\n[green]Total: {stats['total_papers']} papers, {stats['total_chunks']} chunks[/green]")


@app.command()
def stats():
    """
    Show system statistics.
    """
    vector_store = VectorStore()
    settings = get_settings()
    stats = vector_store.get_stats()

    table = Table(title="System Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Papers", str(stats["total_papers"]))
    table.add_row("Total Chunks", str(stats["total_chunks"]))
    table.add_row("LLM Provider", settings.llm_provider)
    table.add_row("LLM Model", settings.llm_model)
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("Chunk Size", str(settings.chunk_size))

    console.print(table)


@app.command()
def delete(
    paper_title: str = typer.Argument(..., help="Title of the paper to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Delete a paper from the index.
    """
    if not force:
        confirm = typer.confirm(f"Delete paper '{paper_title}'?")
        if not confirm:
            raise typer.Abort()

    vector_store = VectorStore()
    deleted = vector_store.delete_paper(paper_title)

    if deleted > 0:
        console.print(f"[green]Deleted {deleted} chunks for '{paper_title}'[/green]")
    else:
        console.print(f"[yellow]No chunks found for '{paper_title}'[/yellow]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """
    Start the REST API server.
    """
    import uvicorn

    console.print(f"[cyan]Starting API server at http://{host}:{port}[/cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def interactive():
    """
    Start an interactive chat session.
    """
    engine = get_engine()

    console.print(Panel(
        "[bold cyan]Research Paper RAG - Interactive Mode[/bold cyan]\n\n"
        "Ask questions about your indexed research papers.\n"
        "Commands: /list, /stats, /quit",
        border_style="cyan",
    ))

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()

            if not question:
                continue

            if question.lower() in ["/quit", "/exit", "/q"]:
                console.print("[dim]Goodbye![/dim]")
                break

            if question.lower() == "/list":
                papers = engine.get_available_papers()
                if papers:
                    for p in papers:
                        console.print(f"  • {p}")
                else:
                    console.print("[yellow]No papers indexed[/yellow]")
                continue

            if question.lower() == "/stats":
                stats()
                continue

            console.print("\n[bold green]Assistant:[/bold green]")
            for chunk in engine.ask_stream(question):
                console.print(chunk, end="")
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

