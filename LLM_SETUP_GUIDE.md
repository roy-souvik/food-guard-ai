# LLM Configuration Guide for FoodGuard AI

## Overview
The agents have been updated to use a **vLLM server** running **Qwen3-4B** with OpenAI-compatible API, replacing the previous Ollama setup.

## Server Setup

### Start the vLLM Server

Run this command in a terminal:

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --api-key abc-123 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --max_model_len 24272
```

The server will be available at `http://localhost:8000/v1`

## Python Integration

### Option 1: Environment Variables (Recommended)

Set up environment variables and use the helper function:

```python
import os
from agents.llm_client import configure_llm_env, get_llm_client

# Configure environment
configure_llm_env(
    base_url="http://localhost:8000/v1",
    api_key="abc-123"
)

# Get LLM client
llm = get_llm_client()
```

### Option 2: Direct Usage in Notebooks

```python
import sys
sys.path.insert(0, '/home/souvik/projects/AI/food-guard-ai')

# Run setup helper
from notebooks.setup_llm import setup_llm
llm = setup_llm()

# Use LLM
from langchain_core.messages import HumanMessage
response = llm.invoke([HumanMessage(content="Your prompt here")])
print(response.content)
```

## Updated Agents

The following agents now use the vLLM server with Qwen3-4B:

### 1. **Food Safety Agent** (`agents/food_safety.py`)
- **Role**: Generates food safety verdicts using multimodal analysis
- **Input**: Vision, Aroma, Taste, and Correlation results
- **Output**: Authenticity score, risk level, verdict, suspected adulterant
- **LLM Task**: Reason over multimodal signals to generate verdict

### 2. **Reviewer Agent** (`agents/reviewer.py`)
- **Role**: Validates verdict and adjusts confidence
- **Input**: All analysis results + initial investigation verdict
- **Output**: Final confidence score, adjustment, escalation flags
- **LLM Task**: Challenge assumptions and identify edge cases

### 3. **Correlation Agent** (`agents/correlation.py`)
- **Role**: Cross-modal evidence fusion
- **Input**: Vision, Aroma, Taste results
- **Output**: Pattern type, matched rules, confidence delta
- **LLM Task**: Reason over multimodal signals to detect adulterants

## Client Configuration

The `llm_client.py` module provides a centralized LLM setup:

```python
from agents.llm_client import get_llm_client, configure_llm_env

# Configure (only needed once per session)
configure_llm_env(
    base_url="http://localhost:8000/v1",
    api_key="abc-123"
)

# Get client (can call multiple times)
llm = get_llm_client()

# Use with LangChain
from langchain_core.messages import HumanMessage
response = llm.invoke([HumanMessage(content="prompt")])
```

## LLM Response Formats

Each agent expects structured JSON responses from the LLM:

### Food Safety Response
```json
{
    "authenticity_score": 85,
    "risk_level": "Low",
    "verdict": "Authentic",
    "suspected_adulterant": "unknown",
    "confidence": 95
}
```

### Reviewer Response
```json
{
    "confidence_adjustment": 5,
    "consensus_level": "High",
    "has_contradictions": false,
    "escalation_needed": false,
    "notes": ["strong cross-modal consensus"],
    "reasoning": "All modalities agree on authenticity"
}
```

### Correlation Response
```json
{
    "pattern_type": "Urea",
    "matched_rules": ["ammonia_spike", "crystal_pattern"],
    "suspected_adulterant": "Urea",
    "confidence_delta": 15,
    "contradictions": [],
    "consensus_score": 92,
    "reasoning": "E-Nose detected ammonia, vision shows crystals"
}
```

## Troubleshooting

### Connection Error
**Problem**: `Error: Connection refused on http://localhost:8000/v1`

**Solution**:
1. Check if vLLM server is running
2. Verify port 8000 is correct
3. Check API key matches (default: `abc-123`)

### Response Parsing Error
**Problem**: LLM response doesn't contain valid JSON

**Solution**:
1. Check LLM output format matches expected schema
2. Verify prompt template includes JSON format instructions
3. Try increasing `max_tokens` in `llm_client.py`

### Memory Issues
**Problem**: `CUDA out of memory` or slow responses

**Solution**:
1. Reduce `max_model_len` in vLLM startup command
2. Enable quantization in vLLM
3. Use smaller batch sizes

## Next Steps

1. ✅ Start vLLM server with Qwen3-4B
2. ✅ Test LLM connection using `setup_llm.py`
3. ✅ Run agent workflows through the graph
4. ✅ Monitor logs for agent execution
5. ✅ Validate verdicts with test data

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3-4B Model](https://huggingface.co/Qwen/Qwen3-4B)
- [LangChain ChatOpenAI](https://python.langchain.com/docs/integrations/llms/openai)
