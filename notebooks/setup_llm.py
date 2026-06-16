"""
LLM Setup Helper for Food Guard AI
Run this at the beginning of any notebook to configure the vLLM server connection
"""

import os
import sys

# Add project to path if running from notebooks folder
sys.path.insert(0, '..')

from llm_client import configure_llm_env, get_llm_client


def setup_llm(base_url: str = "http://localhost:8000/v1", api_key: str = "abc-123"):
    """
    Configure LLM environment and test connection.

    Args:
        base_url: vLLM server URL (default: http://localhost:8000/v1)
        api_key: API key for vLLM server (default: abc-123)
    """
    print("🚀 Setting up FoodGuard AI LLM configuration...\n")

    # Configure environment
    configure_llm_env(base_url, api_key)

    # Get client
    llm = get_llm_client()

    # Test connection
    print("🧪 Testing LLM connection...")
    try:
        from langchain_core.messages import HumanMessage
        test_message = HumanMessage(content="Respond with 'OK' if you receive this.")
        response = llm.invoke([test_message])
        print(f"✅ LLM Connection successful!")
        print(f"   Model: Qwen3-4B")
        print(f"   Server: {base_url}")
        print(f"   Response: {response.content[:100]}...\n")
        return llm
    except Exception as e:
        print(f"❌ LLM Connection failed: {e}")
        print(f"   Make sure vLLM server is running at {base_url}")
        print(f"   Start with: VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve Qwen/Qwen3-4B --served-model-name Qwen3-4B --api-key abc-123 --port 8000\n")
        return None


if __name__ == "__main__":
    llm = setup_llm()
