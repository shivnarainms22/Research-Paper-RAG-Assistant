"""
LLM chat interface supporting multiple providers.
"""

from dataclasses import dataclass
from typing import Generator, Optional

from openai import OpenAI
from anthropic import Anthropic

from src.config import get_settings


@dataclass
class ChatMessage:
    """A message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from the LLM."""

    content: str
    model: str
    tokens_used: int
    finish_reason: str


class ChatLLM:
    """
    Unified interface for chat-based LLMs (OpenAI, Anthropic).
    
    Example:
        llm = ChatLLM()
        response = llm.chat("Explain reinforcement learning")
        
        # With system prompt
        response = llm.chat(
            "What is PPO?",
            system_prompt="You are an RL expert."
        )
        
        # Streaming
        for chunk in llm.stream("Explain attention mechanisms"):
            print(chunk, end="")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the chat LLM.
        
        Args:
            provider: LLM provider ('openai' or 'anthropic').
            model: Model name to use.
        """
        settings = get_settings()
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model

        if self.provider == "openai":
            self.client = OpenAI(api_key=settings.openai_api_key)
        elif self.provider == "anthropic":
            self.client = Anthropic(api_key=settings.anthropic_api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> ChatResponse:
        """
        Send a message and get a response.
        
        Args:
            message: The user's message.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in response.
            
        Returns:
            ChatResponse object.
        """
        if self.provider == "openai":
            return self._chat_openai(message, system_prompt, temperature, max_tokens)
        else:
            return self._chat_anthropic(message, system_prompt, temperature, max_tokens)

    def _chat_openai(
        self,
        message: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        """Chat using OpenAI API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return ChatResponse(
            content=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            finish_reason=response.choices[0].finish_reason,
        )

    def _chat_anthropic(
        self,
        message: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        """Chat using Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": message}],
        )

        content = response.content[0].text if response.content else ""

        return ChatResponse(
            content=content,
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )

    def stream(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> Generator[str, None, None]:
        """
        Stream a response token by token.
        
        Args:
            message: The user's message.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            
        Yields:
            Response text chunks.
        """
        if self.provider == "openai":
            yield from self._stream_openai(message, system_prompt, temperature, max_tokens)
        else:
            yield from self._stream_anthropic(message, system_prompt, temperature, max_tokens)

    def _stream_openai(
        self,
        message: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        """Stream from OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _stream_anthropic(
        self,
        message: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        """Stream from Anthropic."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

