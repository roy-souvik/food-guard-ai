# FoodGuard AI — Complete Technical Plan

**Project**: Multi-Agent Milk Authenticity Investigation Platform

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technology Stack](#technology-stack)
3. [Database Schema](#database-schema)
4. [LangGraph Agents](#langgraph-agents)
5. [Correlation Engine](#correlation-engine)
6. [LLM Integration](#llm-integration)
7. [Synthetic Data Strategy](#synthetic-data-strategy)
8. [Notebooks Specification](#notebooks-specification)
9. [Execution Flow](#execution-flow)
10. [Quick Reference](#quick-reference)

---

## Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                  Gradio UI (Notebook 08)                    │
│          4 Tabs: Input | Processing | Reasoning | Verdict   │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼─────────┐  ┌──────────────────────┐
│ Correlation     │  │  Food Safety         │
│ Agent (05)      │  │  Agent (06)          │
│ LangGraph 5-node│  │  LangGraph 4-node    │
└───────┬─────────┘  └──────────┬───────────┘
        │                       │
        └───────────┬───────────┘
                    │
        ┌───────────▼────────────┐
        │   SQLite Database      │
        │   (11 tables)          │
        │                        │
        │  - investigations      │
        │  - evidence            │
        │  - agent_execution     │
        │  - correlation_rules   │
        │  - passports           │
        │  - model_registry      │
        └────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph** | State management, clear nodes/edges, deterministic execution |
| **SQLite** | Fast, lightweight, good for demos, full ACID compliance |
| **Ollama Qwen3** | Local LLM, no API calls needed, privacy-preserving |
| **Weighted Voting** | Transparent, interpretable, matches agricultural research |
| **Synthetic Data** | No licensing issues, reproducible, realistic distributions |
| **Gradio UI** | Fast to build, minimal dependencies, good for hackathon |

---

## Technology Stack

### Core Libraries

```yaml
Orchestration:
  - langgraph: ^0.0.21          # Multi-agent workflows
  - langchain: ^0.1.0            # LLM abstractions

UI:
  - gradio: ^3.50.0              # Interactive demo interface
  - streamlit: (alternative)     # Not using, but viable

Data:
  - sqlite3: (stdlib)            # Embedded database
  - pandas: ^1.5.0               # Data manipulation
  - numpy: ^1.23.0               # Numerical computing

ML:
  - xgboost: (optional)          # For real model training
  - tensorflow: (optional)       # For vision models

Utilities:
  - qrcode: ^7.4.2               # QR code generation
  - pillow: ^9.0.0               # Image processing
  - requests: (stdlib)           # HTTP for Ollama
```

### External Services

```yaml
Ollama (LLM Inference):
  - Download: https://ollama.ai
  - Model: qwen3
  - Endpoint: http://localhost:11434
  - Status: Start with `ollama serve`

Database:
  - Type: SQLite3
  - Location: ./foodguard.db
  - Size: ~50 MB (after data generation)
  - Backup: Copy foodguard.db file
```

---

## Database Schema

### 11 Tables (Complete Design)

#### 1. investigations
```sql
CREATE TABLE investigations (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    predicted_class TEXT,
    confidence REAL,
    risk_level TEXT,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(batch_id) REFERENCES aroma_analysis(batch_id)
)
```
**Purpose**: Main investigation record, links evidence to verdict

#### 2. evidence
```sql
CREATE TABLE evidence (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    evidence_type TEXT,          -- 'aroma', 'taste', 'vision'
    confidence REAL,
    metadata JSON,               -- Raw sensor data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(investigation_id) REFERENCES investigations(id)
)
```
**Purpose**: Stores raw sensor readings for each investigation

#### 3. aroma_analysis
```sql
CREATE TABLE aroma_analysis (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    ammonia REAL, alcohol REAL, voc REAL, sulfur REAL, hydrocarbon REAL,
    predicted_class TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```
**Purpose**: E-Nose predictions (5 sensors, 7 classes)

#### 4. taste_analysis
```sql
CREATE TABLE taste_analysis (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    sweetness REAL, saltiness REAL, sourness REAL,
    bitterness REAL, umami REAL, astringency REAL,
    predicted_class TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```
**Purpose**: E-Tongue predictions (6 sensors, 7 classes)

#### 5. vision_analysis
```sql
CREATE TABLE vision_analysis (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    image_path TEXT,
    deposit_type TEXT,
    predicted_class TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```
**Purpose**: Vision model predictions (deposit patterns, 7 classes)

#### 6. correlation_rules
```sql
CREATE TABLE correlation_rules (
    id TEXT PRIMARY KEY,
    target_adulterant TEXT,      -- 'Urea', 'Formalin', etc.
    pattern_json JSON,           -- {aroma: [...], taste: [...], vision: [...]}
    weight REAL,                 -- 0.5-0.95
    explanation TEXT,            -- Human-readable rule
    confidence_boost REAL DEFAULT 0.1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```
**Purpose**: Research-backed correlation patterns, used for voting

#### 7. correlations
```sql
CREATE TABLE correlations (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    matched_rules JSON,          -- [{rule_id, weight, match_count}, ...]
    candidate_adulterants JSON,  -- [{class: 'Urea', score: 0.85}, ...]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(investigation_id) REFERENCES investigations(id)
)
```
**Purpose**: Stores matched rules and voting results per investigation

#### 8. agent_execution
```sql
CREATE TABLE agent_execution (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    agent_name TEXT,             -- 'CorrelationAgent', 'FoodSafetyAgent'
    reasoning TEXT,              -- Full LLM response or agent logic
    execution_time_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(investigation_id) REFERENCES investigations(id)
)
```
**Purpose**: Full audit trail of agent decisions

#### 9. passports
```sql
CREATE TABLE passports (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    passport_json JSON,          -- Complete investigation data
    hash_sha256 TEXT UNIQUE,     -- Integrity verification
    qr_code_path TEXT,           -- Path to PNG QR code
    blockchain_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(investigation_id) REFERENCES investigations(id)
)
```
**Purpose**: Cryptographically signed investigation records

#### 10. reports
```sql
CREATE TABLE reports (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    report_markdown TEXT,        -- Full Markdown report
    report_json JSON,            -- Structured data
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(investigation_id) REFERENCES investigations(id)
)
```
**Purpose**: Human-readable and machine-readable reports

#### 11. model_registry
```sql
CREATE TABLE model_registry (
    id TEXT PRIMARY KEY,
    model_name TEXT,             -- 'aroma_model', 'taste_model', 'vision_model'
    version TEXT,
    model_path TEXT,             -- Path to saved model artifact
    accuracy REAL,
    trained_samples INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```
**Purpose**: Track ML model versions and performance

---

## LangGraph Agents

### Agent 1: CorrelationAgent (Notebook 05)

**Purpose**: Apply weighted voting to sensor predictions

**Architecture**: 5-node LangGraph StateGraph

```
┌─────────────────┐
│  Input State    │
│  - batch_id     │
│  - 3 predictions│
└────────┬────────┘
         │
    ┌────▼──────────────────┐
    │ normalize_signals      │ Convert confidence → signal tokens (high/medium/low)
    └────┬───────────────────┘
         │
    ┌────▼──────────────────┐
    │ load_rules            │ Fetch all correlation_rules from DB
    └────┬───────────────────┘
         │
    ┌────▼──────────────────┐
    │ match_rules           │ Find rules where ≥1 signal matches
    │                        │ Sort by: match_count DESC, weight DESC
    └────┬───────────────────┘
         │
    ┌────▼──────────────────┐
    │ weighted_vote         │ Score = (0.4×aroma + 0.4×taste + 0.2×vision)
    │                        │        × rule_weight × match_count
    │                        │ Aggregate by adulterant, normalize
    └────┬───────────────────┘
         │
    ┌────▼──────────────────┐
    │ finalize              │ Create investigation record, log to agent_execution
    │                        │ Return investigation_id, execution_id
    └────┬───────────────────┘
         │
    └─── [END] ──┘
```

**State TypedDict**:
```python
class CorrelationAgentState(TypedDict):
    batch_id: str
    aroma_pred: dict           # {class, confidence, signals}
    taste_pred: dict
    vision_pred: dict
    aroma_signal: str          # 'high', 'medium', 'low'
    taste_signal: str
    vision_signal: str
    matched_rules: list        # [{rule_id, weight, match_count}, ...]
    candidate_adulterants: list # [{class, score}, ...]
    selected_adulterant: str
    final_confidence: float
    investigation_id: str
    execution_id: str
    execution_time_ms: float
```

**Key Method**: `invoke(batch_id: str) -> FinalState`
1. Fetch aroma/taste/vision from DB for batch_id
2. Build initial_state
3. Execute compiled_graph.invoke(initial_state)
4. Return with investigation_id (persisted to DB)

**Test Case**:
```python
# Input: Fresh milk sample (batch_id="batch_001")
# E-Nose: Fresh (0.85)
# E-Tongue: Fresh (0.80)
# Vision: clean_ring (0.90)

# Expected Output:
# - matched_rules: 3 rules (Fresh Clean Ring Triple, Fresh Sweet Balance, etc.)
# - selected_adulterant: 'Fresh'
# - final_confidence: 0.85
```

---

### Agent 2: FoodSafetyAgent (Notebook 06)

**Purpose**: Apply LLM reasoning to investigation + assign risk level

**Architecture**: 4-node LangGraph StateGraph

```
┌──────────────────┐
│  Input State     │
│  - investigation │
│  - predictions   │
└────────┬─────────┘
         │
    ┌────▼──────────────────┐
    │ build_prompt          │ Construct structured prompt with all evidence
    └────┬───────────────────┘
         │
    ┌────▼──────────────────┐
    │ call_llm              │ Call ollama_client.generate() with Qwen3
    │                        │ Fallback template if unavailable
    └────┬───────────────────┘
         │
    ┌────▼──────────────────┐
    │ parse_response        │ Extract RISK LEVEL, reasoning, recommendation
    └────┬───────────────────┘
         │
    ┌────▼──────────────────┐
    │ finalize              │ Update investigations table, log to agent_execution
    └────┬───────────────────┘
         │
    └─── [END] ──┘
```

**Prompt Template**:
```
[INVESTIGATION ID]: {investigation_id}

[BATCH SENSORS]:
- E-Nose (Aroma): {aroma_class} ({aroma_conf:.0%})
- E-Tongue (Taste): {taste_class} ({taste_conf:.0%})
- Vision (Deposit): {vision_class} ({vision_conf:.0%})

[CORRELATION VERDICT]:
- Suspected Adulterant: {predicted_class}
- Confidence: {confidence:.0%}
- Matched Rules: {matched_rules_count}

[TASK]:
Analyze the evidence and provide:
1. RISK LEVEL: Low / Medium / High / Critical
2. Key evidence analysis
3. Recommendation for milk handling

[FORMAT]:
RISK LEVEL: [High]
REASONING: [Your detailed analysis...]
RECOMMENDATION: [Action to take...]
```

**Fallback Reasoning** (if Ollama unavailable):
```python
fallback_responses = {
    "Fresh": ("Low", "Consistent signals, no adulterant markers, safe for consumption"),
    "Water": ("Medium", "Weak all signals indicate dilution, nutritional value reduced"),
    "Urea": ("High", "Ammonia + bitterness + crystals classic triplet, health risk"),
    "Detergent": ("High", "Needle crystals + harsh taste, chemical contamination"),
    "Formalin": ("Critical", "Pungent formaldehyde smell, dark deposit, HIGH TOXICITY"),
    "H2O2": ("High", "Oxidative markers, bleached taste, potential toxin"),
    "Spoiled": ("Critical", "Sour aroma/taste, microbial fermentation, UNSAFE")
}
```

**State TypedDict**:
```python
class FoodSafetyAgentState(TypedDict):
    investigation_id: str
    batch_id: str
    suspected_adulterant: str
    correlation_confidence: float
    aroma_pred: dict
    taste_pred: dict
    vision_pred: dict
    matched_rules: int
    llm_response: str
    risk_level: str           # Low/Medium/High/Critical
    reasoning: str
    final_verdict: str
    execution_id: str
    execution_time_ms: float
```

**Key Method**: `invoke(investigation_id: str) -> FinalState`
1. Fetch investigation from DB
2. Fetch all predictions
3. Build prompt
4. Call Ollama (or fallback)
5. Parse risk level
6. Update investigation record (persisted to DB)
7. Log to agent_execution
8. Return final_state

**Test Case**:
```python
# Input: Formalin-adulterated milk investigation
# From previous CorrelationAgent output
# investigation_id="inv_001"

# Expected Output (with Qwen3):
# risk_level: "Critical"
# reasoning: "[LLM explanation about formaldehyde markers]"

# Expected Output (fallback):
# risk_level: "Critical"
# reasoning: "Pungent formaldehyde smell, dark deposit, HIGH TOXICITY"
```

---

## Correlation Engine

### Weighted Voting Mechanism

**Formula**:
```
For each adulterant candidate:
  score = Σ(for each matched_rule:
    (0.4 × aroma_confidence + 0.4 × taste_confidence + 0.2 × vision_confidence)
    × rule_weight
    × match_count
  )

  final_confidence = normalize(score) to [0, 1]
```

**Example Calculation**:

```
Inputs:
- aroma_confidence: 0.85 (Urea)
- taste_confidence: 0.80 (Urea)
- vision_confidence: 0.90 (fine_crystals matches Urea)

Matched Rules:
1. "Urea Classic Triple" (weight=0.95, match_count=3)
2. "High Ammonia Signal" (weight=0.85, match_count=1)

Calculation:
  rule_1_score = (0.4×0.85 + 0.4×0.80 + 0.2×0.90) × 0.95 × 3
               = (0.34 + 0.32 + 0.18) × 0.95 × 3
               = 0.84 × 0.95 × 3
               = 2.394

  rule_2_score = (0.4×0.85 + 0.4×0.80 + 0.2×0.90) × 0.85 × 1
               = 0.84 × 0.85 × 1
               = 0.714

  total_score = 2.394 + 0.714 = 3.108
  normalized = 3.108 / 4.0 = 0.777 (77.7% confidence)

Final Verdict: Urea (77.7%)
```

### Correlation Rules (20+ Examples)

**Rules stored in DB** (pattern_json field):

```json
{
  "id": "rule_001",
  "target_adulterant": "Urea",
  "pattern_json": {
    "aroma": ["high_ammonia"],
    "taste": ["high_bitterness"],
    "vision": ["fine_crystals"]
  },
  "weight": 0.95,
  "explanation": "Classic Urea signature: ammonia + bitterness + fine dispersed crystals"
}
```

**Full Rule Set**:

| # | Adulterant | Aroma | Taste | Vision | Weight |
|----|-----------|-------|-------|--------|--------|
| 1 | Fresh | high | high | clean_ring | 0.95 |
| 2 | Fresh | sweet | sweet | - | 0.80 |
| 3 | Fresh | - | - | clean_ring | 0.70 |
| 4 | Water | weak | weak | weak_ring | 0.85 |
| 5 | Water | diluted | diluted | - | 0.75 |
| 6 | Urea | high_ammonia | high_bitterness | fine_crystals | 0.95 |
| 7 | Urea | high_ammonia | - | - | 0.70 |
| 8 | Urea | - | high_bitterness | - | 0.60 |
| 9 | Detergent | harsh | harsh_taste | needle_crystals | 0.90 |
| 10 | Detergent | - | astringent | needle_crystals | 0.70 |
| 11 | Formalin | pungent | sour | dark_deposit | 0.95 |
| 12 | Formalin | alcohol_voc | - | - | 0.65 |
| 13 | H2O2 | - | bleached | - | 0.60 |
| 14 | H2O2 | bleach_like | - | - | 0.65 |
| 15 | Spoiled | sour_ammonia | sour | dark_deposit | 0.95 |
| 16 | Spoiled | fermented | fermented | - | 0.80 |
| 17 | Spoiled | - | - | dark_deposit | 0.55 |
| 18 | Melamine | - | bitter_metallic | - | 0.70 |
| 19 | Melamine | - | - | crystalline | 0.60 |
| 20 | Starch | - | - | starch_crystals | 0.65 |

---

## LLM Integration

### Ollama Configuration

**Setup**:
```bash
# 1. Download Ollama from https://ollama.ai
# 2. Pull model
ollama pull qwen3

# 3. Start server
ollama serve
# Runs on http://localhost:11434
```

**Python Wrapper** (in foodguard_lib.py):
```python
class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def generate(self, model: str, prompt: str, stream=False) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": stream},
            timeout=30
        )
        return response.json()["response"]
```

**Prompt Engineering**:
```
Key principles:
1. Structured sections (INVESTIGATION ID, BATCH SENSORS, etc.)
2. Explicit format requirements (RISK LEVEL: [value])
3. Evidence-first reasoning
4. Action-oriented recommendations

Temperature: 0.7 (deterministic but not robotic)
Max tokens: 500
```

**Fallback Strategy**:
```python
def _fallback_reasoning(state):
    """Template-based reasoning if Ollama unavailable"""
    fallback_map = {
        "Fresh": ("Low", "consistent signals, no adulterant markers"),
        "Water": ("Medium", "weak all signals indicate dilution"),
        # ... (7 total mappings)
    }
    risk, reasoning = fallback_map.get(state["suspected_adulterant"], ("Unknown", ""))
    return state | {"risk_level": risk, "reasoning": reasoning}
```

---

## Synthetic Data Strategy

### E-Nose Dataset (7000 samples)

**Characteristics**:
- 7 classes × 1000 samples each
- 5 sensors: ammonia, alcohol, VOC, sulfur, hydrocarbon
- Gaussian noise around literature-backed ranges

**Per-Class Profiles**:

| Class | Ammonia | Alcohol | VOC | Sulfur | Hydrocarbon |
|-------|---------|---------|-----|--------|------------|
| Fresh | 0.1-0.5 | 0.1-0.4 | 0.2-0.6 | 0.05-0.3 | 0.1-0.4 |
| Water | 0.05-0.2 | 0.05-0.15 | 0.1-0.3 | 0.02-0.1 | 0.05-0.2 |
| Urea | 1.5-3.0 | 0.1-0.4 | 0.5-1.5 | 0.2-0.8 | 0.3-1.0 |
| Detergent | 0.2-0.8 | 0.3-1.2 | 2.0-4.0 | 0.5-1.5 | 1.0-2.0 |
| Formalin | 0.5-1.5 | 1.5-3.5 | 2.0-4.5 | 0.3-1.0 | 0.5-1.5 |
| H2O2 | 0.1-0.4 | 0.3-1.0 | 0.8-1.8 | 0.2-0.6 | 0.2-0.8 |
| Spoiled | 1.0-2.5 | 0.5-1.5 | 1.5-3.5 | 0.5-1.5 | 0.8-2.0 |

**Mock Predictions**:
- 85% of samples: Correct class with 0.7-0.95 confidence
- 15% of samples: Random misclassification with 0.4-0.65 confidence

### E-Tongue Dataset (7000 samples)

**Characteristics**:
- 7 classes × 1000 samples each
- 6 sensors: sweetness, saltiness, sourness, bitterness, umami, astringency
- Range: 0.0-5.0 (normalized intensity)

**Per-Class Profiles**:

| Class | Sweetness | Saltiness | Sourness | Bitterness | Umami | Astringency |
|-------|-----------|-----------|----------|-----------|-------|-------------|
| Fresh | 4.0-6.0 | 2.0-3.5 | 0.5-1.5 | 0.2-1.0 | 3.0-4.0 | 0.1-0.5 |
| Water | 1.0-2.0 | 0.5-1.5 | 0.2-0.8 | 0.1-0.5 | 0.5-1.5 | 0.0-0.3 |
| Urea | 1.0-2.5 | 1.5-3.0 | 0.8-2.0 | 2.0-4.0 | 1.5-3.0 | 0.3-1.5 |
| Detergent | 0.5-1.5 | 3.0-4.5 | 2.0-3.5 | 3.0-5.0 | 1.0-2.5 | 2.0-4.0 |
| Formalin | 0.2-1.0 | 1.0-2.5 | 3.0-5.0 | 1.5-3.5 | 0.8-2.0 | 1.0-3.0 |
| H2O2 | 1.0-2.5 | 2.0-3.5 | 2.5-4.0 | 0.5-2.0 | 0.8-2.0 | 0.5-1.5 |
| Spoiled | 0.5-2.0 | 1.5-3.0 | 3.5-5.0 | 1.0-3.0 | 2.0-4.0 | 1.5-3.5 |

### Vision Dataset (420 images)

**Characteristics**:
- 7 classes × 60 images each
- 256×256 RGB procedurally generated using PIL
- Specific patterns per adulterant

**Pattern Types**:
1. **Fresh**: Clean concentric rings (white + light blue)
2. **Water**: Weak ring (pale, fuzzy edges)
3. **Urea**: Fine dispersed crystals (small white dots)
4. **Detergent**: Needle crystals (elongated white crystals)
5. **Formalin**: Dark deposits (brown/black particles)
6. **H2O2**: Bleached appearance (very light, washed out)
7. **Spoiled**: Dark deposits + cracks (brown with fracture patterns)

**Generation Code**:
```python
def generate_vision_image(adulterant_class, image_size=256):
    img = Image.new('RGB', (image_size, image_size), color='white')
    draw = ImageDraw.Draw(img)

    if adulterant_class == "Fresh":
        for i in range(50, 200, 30):
            draw.ellipse([i, i, image_size-i, image_size-i], outline='blue')
    elif adulterant_class == "Urea":
        for _ in range(200):
            x = random.randint(0, image_size)
            y = random.randint(0, image_size)
            draw.point((x, y), fill='gray')
    # ... (5 more classes)

    return img
```

---

## Notebooks Specification

### Notebook 00: setup_db.ipynb

**Execution Time**: ~2 minutes

**Cells**:
1. Imports & initialization
2. Directory creation (./data, ./certificates, ./models, ./reports)
3. `init_db()` → Creates 11 tables
4. Define ~20 correlation rules
5. `insert_correlation_rule()` for each
6. Verification queries
7. Summary stats (rules by adulterant)

**Outputs**:
- foodguard.db created (empty)
- 11 tables initialized
- 20+ rules inserted
- Verification log

---

### Notebook 01: synthetic_data_generation.ipynb

**Execution Time**: ~5 minutes

**Cells**:
1. Imports & RNG seeding (seed=42)
2. Define E-Nose generation function
3. Generate 7000 E-Nose samples
4. Define E-Tongue generation function
5. Generate 7000 E-Tongue samples
6. Define vision image generation
7. Generate 420 vision images
8. Define mock prediction logic
9. Insert all into DB (batches)
10. Create demo_samples_mapping.json
11. DB verification queries
12. Sample inspection

**Outputs**:
- aroma_analysis: 7000 rows
- taste_analysis: 7000 rows
- vision_analysis: 7000 rows
- 420 PNG images in data/synthetic/vision/
- demo_samples_mapping.json

---

### Notebook 05: correlation_engine.ipynb

**Execution Time**: ~3 minutes

**Cells**:
1. Imports & LangGraph setup
2. Define CorrelationAgentState TypedDict
3. Define node functions (normalize_signals, load_rules, match_rules, weighted_vote, finalize)
4. Create StateGraph & compile
5. Define `invoke(batch_id)` method
6. Test on 10 synthetic batches
7. Display sample results
8. Verify agent_execution logs

**Outputs**:
- investigations: 10 rows (from tests)
- agent_execution: 10 rows (CorrelationAgent)
- Test results showing verdicts

---

### Notebook 06: food_safety_reasoning.ipynb

**Execution Time**: ~5 minutes

**Cells**:
1. Imports & LangGraph setup
2. Initialize OllamaClient
3. Define FoodSafetyAgentState TypedDict
4. Define node functions (build_prompt, call_llm, parse_response, finalize)
5. Define fallback_reasoning()
6. Create StateGraph & compile
7. Define `invoke(investigation_id)` method
8. Test on 5 investigations
9. Display sample reasoning outputs
10. Check agent_execution logs

**Outputs**:
- investigations: risk_level + reasoning updated
- agent_execution: 5 rows (FoodSafetyAgent)
- Test results showing risk levels

---

### Notebook 07: passport_certificate.ipynb

**Execution Time**: ~3 minutes

**Cells**:
1. Imports & setup
2. Define `generate_investigation_passport()` function
3. Get test investigation
4. Generate passport (JSON + hash + QR)
5. Store in DB
6. Display passport details
7. Test on 5 investigations
8. Verify passports table
9. Display QR code example

**Outputs**:
- passports: 5+ rows
- QR code PNG files in certificates/
- Passport JSON structures in DB

---

### Notebook 08: gradio_demo.ipynb

**Execution Time**: ~1 minute (+ server running)

**Cells**:
1. Imports & setup
2. Define helper functions (get_demo_samples, run_investigation_pipeline)
3. Build Gradio UI with 4 tabs
4. Define click handler
5. Launch demo

**UI Layout**:
- Tab 1: Sample selector dropdown + usage instructions
- Tab 2: Processing output (evidence loaded, agents running)
- Tab 3: Reasoning output (LLM response)
- Tab 4: Verdict output + full markdown report
- Control: "Run Investigation" button

**Outputs**:
- Gradio server on http://localhost:7860
- Interactive investigation results

---

## Execution Flow

### End-to-End Pipeline

```
User selects sample in Gradio UI
           ↓
[Notebook 08] Orchestration
           ↓
Fetch aroma_analysis, taste_analysis, vision_analysis from DB
           ↓
[Notebook 05] CorrelationAgent
  - normalize_signals
  - load_rules
  - match_rules
  - weighted_vote
  - finalize → investigation record created
           ↓
[Notebook 06] FoodSafetyAgent
  - build_prompt
  - call_llm (Ollama Qwen3 or fallback)
  - parse_response
  - finalize → update investigation with risk_level + reasoning
           ↓
[Notebook 07] Generate Passport
  - Assemble investigation + evidence + agent logs
  - SHA-256 hash
  - QR code generation
  - Store in DB
           ↓
Display to user in Gradio UI
  - Tab 2: Processing steps
  - Tab 3: LLM reasoning
  - Tab 4: Final verdict + report
```

### Time Breakdown

| Step | Time |
|------|------|
| DB query (predictions) | ~10ms |
| CorrelationAgent execution | ~50ms |
| FoodSafetyAgent (LLM) | ~2-5s (or fallback ~50ms) |
| Passport generation | ~100ms |
| UI display | ~100ms |
| **TOTAL** | **~2-6s** (or ~200ms with fallback) |

---

## Quick Reference

### Key Files

```
/projects/food-guard-ai/
├── foodguard_lib.py                    (500+ lines, shared utilities)
├── foodguard.db                        (created by notebook 00)
├── notebooks/
│   ├── 00_setup_db.ipynb              (DB + rules)
│   ├── 01_synthetic_data_generation.ipynb (data)
│   ├── 02_train_models.ipynb          (optional stub)
│   ├── 05_correlation_engine.ipynb    (LangGraph agent 1)
│   ├── 06_food_safety_reasoning.ipynb (LangGraph agent 2 + LLM)
│   ├── 07_passport_certificate.ipynb  (reports)
│   └── 08_gradio_demo.ipynb           (UI)
├── data/
│   ├── synthetic/
│   │   └── vision/ (420 images)
│   └── demo_samples_mapping.json
└── certificates/ (QR codes)
```

### Essential Commands

```bash
# Install dependencies
pip install langgraph gradio ollama qrcode pillow numpy pandas

# Start Ollama (separate terminal)
ollama serve
ollama pull qwen3

# Run Jupyter
jupyter notebook

# Access Gradio demo
# http://localhost:7860
```

### Database Queries

```sql
-- Get sample investigation
SELECT * FROM investigations WHERE risk_level = 'Critical' LIMIT 1;

-- Get agent execution logs
SELECT agent_name, execution_time_ms, reasoning
FROM agent_execution
ORDER BY created_at DESC LIMIT 10;

-- Get correlation rules by adulterant
SELECT * FROM correlation_rules
WHERE target_adulterant = 'Urea'
ORDER BY weight DESC;

-- Get passports with hashes
SELECT id, investigation_id, hash_sha256
FROM passports LIMIT 5;
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| No module 'langgraph' | `pip install langgraph langchain` |
| Ollama not accessible | Start `ollama serve` in separate terminal |
| Port 7860 in use | Change in notebook: `demo.launch(server_port=7861)` |
| Database locked | Close other connections or delete foodguard.db |
| No batches found | Run notebook 01 first |

---

## Success Criteria (For Judges)

✅ Multiple agents (CorrelationAgent + FoodSafetyAgent) working together
✅ LangGraph for orchestration with clear state management
✅ ML predictions (mocked but realistic)
✅ Explainable AI (LLM reasoning + correlation rules)
✅ End-to-end traceability (agent_execution logs)
✅ Interactive UI (Gradio with 4 tabs)
✅ Research-backed (correlation rules from literature)
✅ Production-ready code (error handling, logging, documentation)

---
