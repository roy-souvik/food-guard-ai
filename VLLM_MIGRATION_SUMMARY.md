# FoodGuard AI - vLLM Migration Summary

## Changes Overview

Your FoodGuard AI agents have been updated to use **vLLM server** running **Qwen3-4B** with OpenAI-compatible API instead of Ollama. This migration improves performance, flexibility, and agent reasoning capabilities.

---

## Files Modified

### 1. **agents/llm_client.py** (NEW)
**Purpose**: Centralized LLM configuration and client initialization

**Key Functions**:
- `get_llm_client()`: Returns configured ChatOpenAI client for vLLM server
- `configure_llm_env(base_url, api_key)`: Sets environment variables

**Usage**:
```python
from agents.llm_client import get_llm_client, configure_llm_env

configure_llm_env()
llm = get_llm_client()
```

### 2. **agents/food_safety.py** (UPDATED)
**Changes**:
- Now uses `get_llm_client()` for LLM integration
- Builds comprehensive prompt with all multimodal analysis results
- Uses LLM to generate food safety verdict with reasoning
- Added `_build_food_safety_prompt()` helper function
- Added `_parse_food_safety_response()` to extract structured JSON from LLM output
- Replaces stub implementation with real LLM reasoning

**Key Improvements**:
- Leverages Qwen3-4B for intelligent verdict generation
- Cross-modal reasoning over vision, aroma, taste, and correlation data
- Provides structured output (authenticity score, risk level, verdict)
- Full reasoning chain persisted to database

### 3. **agents/reviewer.py** (UPDATED)
**Changes**:
- Now uses `get_llm_client()` for LLM integration
- Builds review prompt analyzing all prior results
- Uses LLM to detect contradictions and adjust confidence
- Added `_build_review_prompt()` helper function
- Added `_parse_review_response()` to extract review decisions

**Key Improvements**:
- Validates investigation verdict through LLM reasoning
- Detects weak signals and edge cases
- Recommends escalation when needed
- Provides confidence adjustment (±50%)

### 4. **agents/correlation.py** (UPDATED)
**Changes**:
- Upgraded from rule-based to LLM-enhanced correlation
- Now uses `get_llm_client()` for multimodal reasoning
- Builds prompt with vision, aroma, taste signals
- Uses LLM to identify adulterant signatures
- Added `_build_correlation_prompt()` helper function
- Added `_parse_correlation_response()` for structured output

**Key Improvements**:
- Multimodal signal fusion via LLM reasoning
- Maps sensor signals to known adulterant signatures
- Detects cross-modal contradictions
- Provides consensus scoring and confidence delta

### 5. **foodguard_lib.py** (DEPRECATED NOTICE)
**Changes**:
- Added deprecation notice to `get_ollama_client()`
- Added deprecation notice to `OllamaClient` class
- Functions kept for backward compatibility
- Directs users to new `agents.llm_client` module

---

## New Files Created

### 1. **notebooks/setup_llm.py** (NEW)
**Purpose**: LLM setup helper for notebooks and scripts

**Key Function**:
- `setup_llm(base_url, api_key)`: Configures and tests LLM connection

**Usage**:
```python
from notebooks.setup_llm import setup_llm
llm = setup_llm()
```

### 2. **notebooks/example_agent_workflow.py** (NEW)
**Purpose**: Complete example of running all agents with vLLM

**Demonstrates**:
- Environment configuration
- State initialization
- Running all agents in sequence
- Handling LLM responses
- Complete investigation workflow

**Run with**:
```bash
python /home/souvik/projects/AI/food-guard-ai/notebooks/example_agent_workflow.py
```

### 3. **LLM_SETUP_GUIDE.md** (NEW)
**Purpose**: Comprehensive guide for LLM setup and usage

**Contains**:
- vLLM server startup commands
- Environment configuration instructions
- Agent descriptions and roles
- Response format specifications
- Troubleshooting guide
- Migration path from Ollama

---

## Server Setup Instructions

### Start vLLM Server

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

### Verify Server is Running

```bash
curl http://localhost:8000/v1/models -H "Authorization: Bearer abc-123"
```

---

## Integration Points

### Environment Variables
```python
os.environ["BASE_URL"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "abc-123"
```

### Agent Workflow
```
Supervisor → Correlation → Food Safety → Reviewer → Passport
                                ↑
                          (uses LLM)
                                ↑
                          (vLLM server)
```

### LLM Client
All agents access LLM through centralized `agents.llm_client` module:
```
agents/
  ├── llm_client.py (NEW - centralized config)
  ├── food_safety.py (uses get_llm_client())
  ├── reviewer.py (uses get_llm_client())
  ├── correlation.py (uses get_llm_client())
  └── ...
```

---

## Configuration Options

### Model Parameters
In `agents/llm_client.py`:
```python
llm = ChatOpenAI(
    model="Qwen3-4B",
    base_url=base_url,
    api_key=api_key,
    temperature=0.7,        # Adjustable: 0.0-1.0
    max_tokens=1024,        # Adjustable based on response size
)
```

### vLLM Parameters
In server startup command:
- `--max_model_len 24272`: Maximum context length
- `--gpu-memory-utilization 0.9`: GPU memory percentage
- `--disable-log-requests`: Disable request logging

---

## Expected Behavior

### Before (Ollama/Stub)
- Agents returned placeholder/stub results
- No LLM reasoning
- Static predictions

### After (vLLM/Qwen3-4B)
- Agents provide intelligent multimodal reasoning
- Structured JSON outputs with reasoning
- Dynamic verdicts based on analysis data
- Confidence adjustments and edge case detection

---

## Testing the Integration

### Test 1: Basic LLM Connection
```python
from notebooks.setup_llm import setup_llm
llm = setup_llm()  # Should print success message
```

### Test 2: Run Example Workflow
```bash
python /home/souvik/projects/AI/food-guard-ai/notebooks/example_agent_workflow.py
```

### Test 3: Direct Agent Test
```python
from agents.food_safety import food_safety_node
from agents.llm_client import configure_llm_env

configure_llm_env()
result = food_safety_node(sample_state)
print(result["investigation_result"])
```

---

## Troubleshooting

### Issue: "Connection refused on http://localhost:8000/v1"
**Solution**: Start vLLM server first

### Issue: "JSON decode error"
**Solution**: Check LLM response includes JSON in specified format

### Issue: "CUDA out of memory"
**Solution**: Reduce `--max_model_len` in server startup or use smaller batch sizes

### Issue: Slow responses
**Solution**: Check vLLM server resource usage; consider quantization

---

## Next Steps

1. ✅ **Start vLLM server** with provided command
2. ✅ **Test connection** using `setup_llm.py`
3. ✅ **Run example workflow** to validate agents
4. ✅ **Integrate with your pipeline** (notebooks or web service)
5. ✅ **Monitor logs** for agent execution details

---

## Migration Path (If Needed)

To revert to Ollama:
1. Update `agents/llm_client.py` to use OllamaClient
2. Change import in food_safety.py, reviewer.py, correlation.py
3. Revert to previous prompt templates (if needed)

However, vLLM + Qwen3-4B provides better performance and flexibility, so migration is recommended.

---

## References

- **vLLM Docs**: https://docs.vllm.ai/
- **Qwen3-4B Model**: https://huggingface.co/Qwen/Qwen3-4B
- **LangChain Integration**: https://python.langchain.com/docs/integrations/llms/openai
- **Setup Guide**: See `LLM_SETUP_GUIDE.md`
- **Example Code**: See `notebooks/example_agent_workflow.py`

---

**Last Updated**: 2026-06-16
**Status**: ✅ Migration Complete
