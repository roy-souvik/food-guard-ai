# Quick Reference: Using vLLM with FoodGuard AI Agents

## Quick Start (< 5 minutes)

### 1. Start vLLM Server (Terminal 1)
```bash
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve Qwen/Qwen3-4B \
  --served-model-name Qwen3-4B \
  --api-key abc-123 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --max_model_len 24272
```

### 2. Run Example Workflow (Terminal 2)
```bash
cd /home/souvik/projects/AI/food-guard-ai
python notebooks/example_agent_workflow.py
```

### 3. Expected Output
```
STEP 1: Configure LLM Environment
✓ LLM configured successfully

STEP 2: Initialize Investigation State
✓ Initial state created for batch: Batch-001

STEP 3: Supervisor Node (ID Generation)
✓ Batch ID: BATCH-XXXXX
✓ Investigation ID: INV-XXXXX

STEP 4: Correlation Agent (Cross-Modal Reasoning)
Correlation Result:
  - Pattern Type: Urea
  - Suspected Adulterant: Urea
  - Confidence Delta: 15
  - Consensus Score: 92

...
✓ Investigation complete!
```

---

## Common Code Snippets

### Setup LLM in Notebook
```python
import sys
sys.path.insert(0, '/home/souvik/projects/AI/food-guard-ai')

from agents.llm_client import configure_llm_env, get_llm_client

configure_llm_env()
llm = get_llm_client()
print("✓ LLM ready")
```

### Call LLM Directly
```python
from langchain_core.messages import HumanMessage

prompt = "Analyze this milk sample for adulteration..."
response = llm.invoke([HumanMessage(content=prompt)])
print(response.content)
```

### Run Food Safety Agent
```python
from agents.food_safety import food_safety_node
from agents.state import FoodInvestigationState

state = FoodInvestigationState(
    food_type="milk",
    batch_name="Batch-001",
    vision_result={"label": "fine_crystals", "score": 0.72},
    aroma_result={"label": "ammonia_spike", "score": 0.79},
    taste_result={"label": "high_salinity", "score": 0.82},
    correlation_result={},
    batch_id="BATCH-123",
    investigation_id="INV-456",
    vision_id="VIS-789",
    aroma_id="ARO-012",
    taste_id="TAS-345",
    correlation_id="CORR-678",
    passport_id="PASS-901",
    certificate_id="CERT-234",
    errors=[]
)

state = food_safety_node(state)
print("Verdict:", state["investigation_result"]["verdict"])
```

### Run Correlation Agent
```python
from agents.correlation import correlation_node

state = correlation_node(state)
print("Pattern:", state["correlation_result"]["pattern_type"])
print("Confidence Delta:", state["correlation_result"]["confidence_delta"])
```

### Run Reviewer Agent
```python
from agents.reviewer import reviewer_node

state = reviewer_node(state)
print("Final Confidence:", state["review_result"]["final_confidence"])
print("Escalation Needed:", state["review_result"]["escalation_needed"])
```

### Run Complete Workflow
```python
from agents.supervisor import supervisor_node
from agents.correlation import correlation_node
from agents.food_safety import food_safety_node
from agents.reviewer import reviewer_node
from agents.passport import passport_node

# Initialize
state = FoodInvestigationState(food_type="milk", batch_name="B1", ...)

# Run all agents
state = supervisor_node(state)
state = correlation_node(state)
state = food_safety_node(state)
state = reviewer_node(state)
state = passport_node(state)

print(f"Investigation complete: {state['investigation_result']['verdict']}")
```

---

## Checking LLM Status

### Verify Server Running
```bash
curl http://localhost:8000/v1/models -H "Authorization: Bearer abc-123"
```

### Check Response (should show):
```json
{"object":"list","data":[{"id":"Qwen3-4B","object":"model"}]}
```

### Python Test
```python
from notebooks.setup_llm import setup_llm
llm = setup_llm()  # Prints success/failure
```

---

## Environment Variables

### In Notebook
```python
import os
os.environ["BASE_URL"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "abc-123"
```

### In Terminal
```bash
export BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="abc-123"
```

### In .env File
```
BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=abc-123
```

---

## Debugging Tips

### Issue: JSON Parsing Error
```python
# Agent response doesn't contain valid JSON
# Check LLM output format in prompts
# Increase max_tokens in llm_client.py
```

### Issue: Timeout Error
```python
# Increase timeout in vLLM server
# Or reduce prompt length
# Check server resource usage: nvidia-smi
```

### Issue: Memory Error
```bash
# Start vLLM with quantization
vllm serve Qwen/Qwen3-4B \
  --quantization awq \
  --port 8000 \
  ...
```

### Issue: Connection Refused
```python
# 1. Check if server is running: ps aux | grep vllm
# 2. Check port 8000: netstat -tuln | grep 8000
# 3. Restart server with correct parameters
```

---

## Performance Tips

### Faster Responses
1. Reduce `max_tokens` in `llm_client.py`
2. Lower `temperature` for more deterministic responses
3. Use quantization in vLLM (`--quantization awq`)

### Better Reasoning
1. Increase `max_tokens` (e.g., 2048)
2. Use `temperature=0.7` for balanced reasoning
3. Include detailed context in prompts

### Memory Optimization
1. Set `--max_model_len 8192` instead of 24272
2. Use `--quantization` for smaller model footprint
3. Batch requests when possible

---

## Files for Reference

| File | Purpose |
|------|---------|
| `agents/llm_client.py` | LLM configuration |
| `agents/food_safety.py` | Verdict generation |
| `agents/reviewer.py` | Confidence validation |
| `agents/correlation.py` | Multimodal reasoning |
| `agents/supervisor.py` | ID generation |
| `agents/passport.py` | Certificate generation |
| `agents/state.py` | State definition |
| `notebooks/example_agent_workflow.py` | Full example |
| `notebooks/setup_llm.py` | Setup helper |
| `LLM_SETUP_GUIDE.md` | Detailed guide |
| `VLLM_MIGRATION_SUMMARY.md` | Migration details |

---

## Key Differences from Ollama

| Aspect | Ollama | vLLM |
|--------|--------|------|
| **Model** | Various | Qwen3-4B (optimized) |
| **API** | Custom | OpenAI-compatible |
| **Speed** | Moderate | Fast |
| **Memory** | Variable | Configurable |
| **Integration** | `OllamaClient` | `ChatOpenAI` |
| **Status** | Deprecated | Recommended ✅ |

---

## Next Steps

1. ✅ Start vLLM server
2. ✅ Run `example_agent_workflow.py`
3. ✅ Test individual agents
4. ✅ Integrate with your notebooks
5. ✅ Monitor performance

**Questions?** See `LLM_SETUP_GUIDE.md` for detailed troubleshooting
