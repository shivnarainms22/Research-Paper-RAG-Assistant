"""
API Module
==========
FastAPI web interface for the Research Paper RAG system.
"""

from .app import app, create_app

__all__ = ["app", "create_app"]

