#!/usr/bin/env python3
"""
Research Paper RAG System
=========================

Main entry point for the application.

Usage:
    python main.py ingest paper.pdf
    python main.py ask "Explain PPO"
    python main.py serve
    python main.py interactive
"""

from src.cli.main import app

if __name__ == "__main__":
    app()

