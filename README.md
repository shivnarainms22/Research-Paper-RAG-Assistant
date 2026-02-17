# Research Paper RAG System

A powerful AI-powered system for ingesting ML/AI research papers and answering deep technical questions with accurate citations.

## Features

- **PDF Ingestion**: Extract text, equations, and structure from research papers
- **Semantic Search**: Find relevant content using vector embeddings
- **Technical Q&A**: Answer complex questions about paper content
- **Math Explanation**: Break down equations into plain English
- **Paper Comparison**: Compare methodologies across multiple papers
- **Exact Citations**: All answers include specific page/section references
- **Multiple Interfaces**: CLI, REST API, and interactive chat

## Example Queries

```
"Explain PPO like I'm new to RL"
"How is DQN different from PPO?"
"What are the limitations of this paper?"
"Summarize the method section in simple terms"
"What does this loss function mean?"
```

## Quick Start

### 1. Clone and Setup

```bash
cd D:\RAG

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
# Required: At least one LLM provider
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Optional: Customize settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Ingest Papers

```bash
# Single paper
python main.py ingest path/to/paper.pdf

# Directory of papers
python main.py ingest ./papers/ --recursive
```

### 4. Ask Questions

```bash
# Command line
python main.py ask "Explain the attention mechanism"

# Interactive mode
python main.py interactive

# Start API server
python main.py serve
```

## Project Structure

```
D:\RAG/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── .env                   # API keys (create from .env.example)
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   │
│   ├── ingestion/         # PDF processing
│   │   ├── __init__.py
│   │   ├── pdf_parser.py  # Extract text from PDFs
│   │   ├── chunker.py     # Split into retrieval chunks
│   │   └── preprocessor.py # Math extraction, text cleanup
│   │
│   ├── retrieval/         # Vector search
│   │   ├── __init__.py
│   │   ├── embeddings.py  # Generate embeddings
│   │   ├── vector_store.py # ChromaDB storage
│   │   └── retriever.py   # High-level retrieval interface
│   │
│   ├── llm/               # Language model integration
│   │   ├── __init__.py
│   │   ├── chat.py        # LLM chat interface
│   │   ├── prompts.py     # Prompt templates
│   │   └── qa_engine.py   # Main Q&A logic
│   │
│   ├── api/               # REST API
│   │   ├── __init__.py
│   │   └── app.py         # FastAPI application
│   │
│   └── cli/               # Command-line interface
│       ├── __init__.py
│       └── main.py        # Typer CLI
│
├── data/
│   ├── papers/            # Store PDFs here
│   ├── processed/         # Processed paper data
│   └── chroma_db/         # Vector store (auto-created)
│
└── tests/
    └── ...
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ingest <path>` | Ingest PDF file(s) into the system |
| `ask "<question>"` | Ask a question about indexed papers |
| `compare "<paper1>" "<paper2>"` | Compare multiple papers |
| `list` | List all indexed papers |
| `stats` | Show system statistics |
| `delete "<paper>"` | Remove a paper from the index |
| `serve` | Start the REST API server |
| `interactive` | Start interactive chat mode |

## REST API Endpoints

Start the server with `python main.py serve`, then:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/papers/upload` | Upload and index a PDF |
| `POST` | `/ask` | Ask a question |
| `POST` | `/compare` | Compare papers |
| `GET` | `/papers` | List indexed papers |
| `DELETE` | `/papers/{title}` | Delete a paper |
| `GET` | `/stats` | Get system statistics |

### Example API Usage

```bash
# Upload a paper
curl -X POST "http://localhost:8000/papers/upload" \
  -F "file=@paper.pdf"

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the main contribution"}'
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider (openai/anthropic) |
| `LLM_MODEL` | `gpt-4-turbo-preview` | Model to use |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `MAX_CONTEXT_TOKENS` | `8000` | Max tokens for context |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PDF       │────▶│  Chunker    │────▶│  Vector     │
│   Parser    │     │             │     │  Store      │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│  QA         │────▶│  Retriever  │
│   Query     │     │  Engine     │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │   LLM       │
                    │   (GPT-4)   │
                    └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Answer +   │
                    │  Citations  │
                    └─────────────┘
```

## License

MIT License - see LICENSE file for details.

