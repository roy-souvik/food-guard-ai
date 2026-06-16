"""
LLM Client Configuration
Centralized setup for vLLM server with OpenAI-compatible API (Qwen3-4B)
"""

import os
from langchain_openai import ChatOpenAI


def get_llm_client():
    """
    Initialize and return ChatOpenAI client configured for vLLM server.

    Expects environment variables:
    - BASE_URL: vLLM server URL (default: http://localhost:8000/v1)
    - OPENAI_API_KEY: API key for vLLM server (default: abc-123)

    Returns:
        ChatOpenAI: Configured LLM client
    """
    base_url = os.environ.get("BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "abc-123")

    llm = ChatOpenAI(
        model="Qwen3-4B",
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
        max_tokens=1024,
    )

    return llm


def configure_llm_env(base_url: str = "http://localhost:8000/v1", api_key: str = "abc-123"):
    """
    Configure environment variables for LLM client.

    Args:
        base_url: vLLM server URL
        api_key: API key for vLLM server
    """
    os.environ["BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"✓ LLM Config set: {base_url}")
