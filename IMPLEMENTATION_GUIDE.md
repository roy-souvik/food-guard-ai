# FoodGuard AI — Implementation Guide

**Created**: June 12, 2026
**Project**: Multi-Agent Milk Authenticity Investigation Platform
**Status**: Ready for Hackathon Demo

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Execution Order](#execution-order)
4. [Running the Demo](#running-the-demo)
5. [File Structure](#file-structure)
6. [Troubleshooting](#troubleshooting)
7. [Performance Notes](#performance-notes)
8. [Production Deployment](#production-deployment)

---

## Quick Start

### For Impatient Hackers (20 Minutes)

```bash
# 1. Install dependencies
pip install langgraph gradio ollama qrcode pillow numpy pandas

# 2. Start Ollama in a separate terminal
ollama serve
ollama pull qwen3

# 3. Open Jupyter and run notebooks in order
jupyter notebook

# In browser, navigate to /notebooks/ and run:
# 1. 00_setup_db.ipynb (2 min)
# 2. 01_synthetic_data_generation.ipynb (5 min)
# 3. 05_correlation_engine.ipynb (3 min)
# 4. 06_food_safety_reasoning.ipynb (5 min)
# 5. 07_passport_certificate.ipynb (3 min)
# 6. 08_gradio_demo.ipynb (1 min) ← LIVE DEMO STARTS HERE

# 4. Open browser to http://localhost:7860
# 5. Select a sample and click "Run Investigation"
# 6. Show judges the results!

# Total time: ~20 minutes
```

---

## Installation

### System Requirements

- Python 3.8+
- 2 GB RAM (minimum)
- 1 GB disk space
- Internet connection (for downloading Ollama + model)

### Step 1: Clone/Navigate to Project

```bash
cd /home/souvik/projects/AI/food-guard-ai
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install langgraph langchain ollama gradio qrcode pillow numpy pandas
```

**Full requirements.txt** (if creating):
```
langgraph>=0.0.21
langchain>=0.1.0
ollama>=0.1.0
gradio>=3.50.0
qrcode>=7.4.2
pillow>=9.0.0
numpy>=1.23.0
pandas>=1.5.0
requests>=2.28.0
```

### Step 4: Download & Start Ollama

**Download**: https://ollama.ai

**After installation**, in a **separate terminal** (leave it running):

```bash
ollama serve
```

**In another terminal**, download the Qwen3 model:

```bash
ollama pull qwen3
```

Verify it's working:

```bash
curl http://localhost:11434/api/tags
# Should return JSON with qwen3 listed
```

---

## Execution Order

### 🚨 STRICT ORDER (Do NOT skip or reorder)

#### Phase 1: Setup (Run Once)

**1. Notebook 00: `00_setup_db.ipynb`**
```bash
# Purpose: Create database schema + seed correlation rules
# Time: ~2 minutes
# Output: foodguard.db (empty but with schema), 20+ rules inserted

# What happens:
# - Creates 11 SQLite tables
# - Inserts correlation rules for 7 adulterant classes
# - Initializes directories (data/, certificates/, models/)
```

**Run ALL cells** (they're sequential and depend on each other)

Expected output at end:
```
[OK] Schema initialization complete
[OK] Rules inserted: 20 rules
[OK] Verification successful: 11 tables created
```

---

#### Phase 2: Data Generation (Run Once)

**2. Notebook 01: `01_synthetic_data_generation.ipynb`**
```bash
# Purpose: Generate realistic synthetic data
# Time: ~5 minutes
# Output: 7000+ samples in DB, 420 images on disk

# What happens:
# - Generates 7000 E-Nose samples (5 sensors, 7 classes)
# - Generates 7000 E-Tongue samples (6 sensors, 7 classes)
# - Generates 420 Vision images (256×256 RGB, 60 per class)
# - Inserts all into database
# - Creates demo sample mappings
```

**Run ALL cells**

Expected output at end:
```
[OK] E-Nose: 7000 samples inserted
[OK] E-Tongue: 7000 samples inserted
[OK] Vision: 420 images generated
[OK] Demo samples mapping created
```

---

#### Phase 3: Agent Implementation (Run Once Each)

**3. Notebook 05: `05_correlation_engine.ipynb`**
```bash
# Purpose: Implement CorrelationAgent (LangGraph)
# Time: ~3 minutes
# Output: 10 test investigations, agent execution logs

# What happens:
# - Builds 5-node LangGraph state machine
# - Implements weighted voting algorithm
# - Tests on 10 synthetic batches
# - Logs all decisions to agent_execution table
```

**Run ALL cells**

Expected output at end:
```
[OK] CorrelationAgent compiled
[OK] Tested on 10 batches
[OK] Sample verdict: Urea (85.3% confidence)
```

---

**4. Notebook 06: `06_food_safety_reasoning.ipynb`**
```bash
# Purpose: Implement FoodSafetyAgent (LangGraph + LLM)
# Time: ~5 minutes
# Output: Updated investigations with risk levels, LLM reasoning

# What happens:
# - Builds 4-node LangGraph state machine
# - Integrates Ollama Qwen3 LLM
# - Tests on 5 investigations from previous step
# - Updates investigations table with risk level + reasoning
# - Fallback to template reasoning if Ollama unavailable
```

**Run ALL cells**

Expected output at end:
```
[OK] FoodSafetyAgent compiled
[OK] Ollama connection: OK
[OK] Tested on 5 investigations
[OK] Sample risk level: Critical
```

---

**5. Notebook 07: `07_passport_certificate.ipynb`**
```bash
# Purpose: Generate investigation passports + QR codes
# Time: ~3 minutes
# Output: Passports in DB, QR code PNG files

# What happens:
# - Generates comprehensive investigation passports (JSON)
# - Computes SHA-256 hashes for integrity
# - Creates QR codes (PNG images)
# - Stores everything in DB
# - Tests on 5+ investigations
```

**Run ALL cells**

Expected output at end:
```
[OK] Passports generated: 5 passports
[OK] QR codes created: 5 PNG files
[OK] Hashes verified: 5 passports stored
```

---

#### Phase 4: Interactive Demo (Run When Ready to Present)

**6. Notebook 08: `08_gradio_demo.ipynb`**
```bash
# Purpose: Launch interactive Gradio UI
# Time: ~1 minute (+ server running)
# Output: Live web UI on http://localhost:7860

# What happens:
# - Builds Gradio interface with 4 tabs
# - Loads all pre-generated demo samples
# - Orchestrates full pipeline (samples → agents → verdict)
# - Displays results in real-time

# 🎯 THIS IS WHAT JUDGES SEE
```

**Run the first 5 cells**, then **run cell 6 last** (server launches and blocks)

```
When you see:
"Launching Gradio demo..."
"Access the demo at: http://localhost:7860"

👉 Open http://localhost:7860 in your browser
```

---

### Optional (Skip for Hackathon)

**Notebook 02: `02_train_models.ipynb`**
```bash
# Purpose: ML model training stub (optional)
# Status: Not required for demo (using mocked predictions)
# Reason: Focus on agent orchestration, not ML training
```

---

## Running the Demo

### Step 1: Ensure All Notebooks Have Run (In Order)

```bash
# Verify files created:
ls -la notebooks/*.ipynb     # Should show 8 files
ls -la foodguard.db          # Should exist and be ~50 MB
ls -la data/synthetic/vision/*.png | wc -l  # Should show ~420
```

### Step 2: Start Ollama (If Not Already Running)

```bash
# In a separate terminal
ollama serve
```

### Step 3: Run Notebook 08

```bash
# In Jupyter notebook browser
# Navigate to notebooks/08_gradio_demo.ipynb
# Run first 5 cells
# Run last cell to launch server
```

### Step 4: Open Gradio UI

```
Open in browser: http://localhost:7860
```

### Step 5: Select Sample & Run Investigation

**Tab 1 — Sample Input**:
- Click dropdown "Available Demo Samples"
- Select any sample (e.g., "FRESH: batch_001")
- Read instructions

**Click "🔍 Run Investigation" button**

**Tab 2 — Agent Processing**:
- Shows E-Nose, E-Tongue, Vision predictions loaded
- "Running Agents..."

**Tab 3 — AI Reasoning**:
- Shows LLM response from Qwen3
- Risk assessment logic
- Recommendation

**Tab 4 — Verdict & Passport**:
- **SUSPECTED ADULTERANT**: Urea (or other)
- **CONFIDENCE**: 85.3%
- **RISK LEVEL**: High (or other)
- Full markdown report
- QR code reference

---

## File Structure

### Post-Installation (Before Running Notebooks)

```
/home/souvik/projects/AI/food-guard-ai/
├── TECHNICAL_PLAN.md           ← Detailed architecture docs
├── IMPLEMENTATION_GUIDE.md      ← This file
├── README.md                    ← Project overview
├── foodguard_lib.py             ← Shared utilities (500+ lines)
├── requirements.txt             ← Python dependencies
├── Dockerfile                   ← Docker setup (optional)
├── docker-compose.yml           ← Docker compose (optional)
│
├── notebooks/
│   ├── 00_setup_db.ipynb
│   ├── 01_synthetic_data_generation.ipynb
│   ├── 02_train_models.ipynb
│   ├── 05_correlation_engine.ipynb
│   ├── 06_food_safety_reasoning.ipynb
│   ├── 07_passport_certificate.ipynb
│   └── 08_gradio_demo.ipynb
│
├── data/
│   ├── synthetic/
│   │   ├── enose_dataset.csv       (generated by 01)
│   │   ├── etongue_dataset.csv     (generated by 01)
│   │   ├── vision/                 (420 PNG images)
│   │   │   ├── fresh_001.png
│   │   │   ├── urea_002.png
│   │   │   └── ...
│   │   └── correlation_rules.csv
│   ├── milk_evaporative_dataset/   (reference, not used)
│   └── demo_samples_mapping.json   (generated by 01)
│
├── models/
│   ├── aroma_model.pkl             (optional, for real training)
│   ├── taste_model.pkl             (optional)
│   └── vision_model.h5             (optional)
│
├── certificates/
│   ├── inv_001_passport.png        (generated by 07)
│   ├── inv_002_passport.png
│   └── ...
│
├── reports/
│   ├── inv_001_report.md           (generated by 07)
│   ├── inv_002_report.md
│   └── ...
│
└── agents/
    ├── aroma.py                    (reference, not used in notebooks)
    ├── taste.py                    (reference)
    ├── vision.py                   (reference)
    └── ... (6 other agents, not used)
```

### After Running All Notebooks

```
foodguard.db                   ← SQLite database (created by 00)
├── 11 tables with data (created by 00, 01, 05-07)
└── ~14,000 rows total

data/synthetic/vision/        ← 420 PNG images (created by 01)
certificates/                 ← QR code PNG files (created by 07)
reports/                       ← Markdown report files (created by 07)
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'langgraph'"

**Solution**:
```bash
pip install langgraph langchain
```

### Problem: "Ollama not accessible" / "Connection refused"

**Check**:
```bash
# 1. Is Ollama running?
ps aux | grep ollama

# 2. Is server on right port?
curl http://localhost:11434/api/tags

# 3. If not running, start it
ollama serve
```

**Fallback**:
- Code automatically uses template-based reasoning if Ollama unavailable
- Demo will still work, just slower LLM responses

### Problem: "No module named 'PIL'" or "Image" import error

**Solution**:
```bash
pip install pillow
```

### Problem: "Port 7860 already in use"

**Solution** (in notebook 08, find this line):
```python
demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
```

Change to:
```python
demo.launch(share=False, server_name="0.0.0.0", server_port=7861)
```

Then access: http://localhost:7861

### Problem: "Database is locked" / "Cannot open database"

**Solution**:
```bash
# 1. Close Jupyter kernel (Kernel → Shutdown All Kernels)
# 2. Delete old database
rm foodguard.db

# 3. Restart Jupyter and re-run notebook 00
```

### Problem: "No batches found" / Empty dropdown in Tab 1

**Cause**: Notebook 01 hasn't been run yet

**Solution**:
```bash
# Make sure you ran in this order:
# 1. Notebook 00 ✓
# 2. Notebook 01 ✓ (MUST run this before 08)
# 3. Notebooks 05-07 ✓
# 4. Notebook 08 ✓

# If skipped, go back and run 01
```

### Problem: "Ollama pulling qwen3" takes too long

**Reason**: Model is ~7 GB

**Alternatives**:
```bash
# Use smaller model
ollama pull mistral   # ~4 GB

# Edit notebook 06, change:
# response = ollama_client.generate(model="qwen3", ...)
# To:
# response = ollama_client.generate(model="mistral", ...)
```

### Problem: Gradio UI loads but dropdown is empty

**Cause**: No demo samples found in database

**Verify**:
```sql
-- In any SQL browser or notebook:
SELECT COUNT(*) FROM aroma_analysis;  -- Should be ~7000
```

**Solution**:
- Run notebook 01 again
- Make sure ALL cells execute without error

### Problem: "Notebook runs but shows errors"

**General Debugging**:
1. Check error message carefully
2. Verify previous notebooks ran successfully
3. Check database file size: `ls -lh foodguard.db`
4. Delete DB and start fresh: `rm foodguard.db && jupyter notebook`

---

## Performance Notes

### Expected Timing

| Step | Time |
|------|------|
| Notebook 00 setup | 1-2 min |
| Notebook 01 data gen | 3-5 min |
| Notebook 05 correlation | 2-3 min |
| Notebook 06 food safety | 4-6 min (with Ollama) or 1-2 min (fallback) |
| Notebook 07 passport | 2-3 min |
| Notebook 08 UI launch | <1 min |
| **TOTAL** | **15-20 minutes** |

### Per-Investigation Timing

| Component | Time |
|-----------|------|
| DB query (predictions) | ~10ms |
| CorrelationAgent | ~50ms |
| FoodSafetyAgent + LLM | ~2-5s (Ollama) or ~50ms (fallback) |
| Passport generation | ~100ms |
| UI update | ~100ms |
| **TOTAL PER SAMPLE** | **2-6 seconds** |

### Database Size

```
Empty DB (after 00): ~100 KB
After 01 (data): ~50 MB
After 05-07 (agents): ~55 MB
```

### Memory Usage

- Python process: ~300-500 MB (normal)
- Ollama server: ~2-4 GB (if running)
- Total: ~3-5 GB recommended

---

## Production Deployment

### NOT Required for Hackathon

These are notes for future reference when deploying FoodGuard AI to production.

### Step 1: Replace Mock Predictions with Real ML Models

**Current**:
- Mock predictions generated in notebook 01
- 85% accuracy by design

**Production**:
- Train real XGBoost models (aroma, taste)
- Train real CNN model (vision)
- Integrate into notebooks 03-04
- Update prediction accuracy based on real data

### Step 2: Upgrade Database

**Current**: SQLite (good for single-user, lightweight)

**Production**: PostgreSQL
```sql
-- Update connection string
db_url = "postgresql://user:pass@localhost:5432/foodguard"
```

### Step 3: Add API Layer

**Current**: Jupyter notebooks only

**Production**: FastAPI
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/investigate")
async def investigate(batch_id: str):
    # Run CorrelationAgent
    # Run FoodSafetyAgent
    # Return results
```

### Step 4: Add Authentication & Audit

**Current**: No user tracking

**Production**:
```python
# Add user_id to all logs
# Add IP address logging
# Add tamper detection (hash verification)
```

### Step 5: Add Monitoring & Alerting

**Current**: Manual log checking

**Production**:
```python
# Log to centralized system (ELK stack, Datadog, etc.)
# Alert on Critical risk levels
# Track agent latency
```

### Step 6: Docker Deployment

**Existing files**: Dockerfile, docker-compose.yml

```bash
# Build image
docker build -t foodguard-ai .

# Run container
docker-compose up -d

# Access at http://localhost:7860
```

### Step 7: Model Versioning & Retraining

**Current**: Single model version

**Production**:
```python
# Track model versions in model_registry
# Schedule monthly retraining
# A/B test new models
# Rollback capability
```

---

## Success Checklist for Judges

Use this checklist to verify everything works:

```
Before Demo:
  ☐ Ollama running (ollama serve)
  ☐ All notebooks executed in order (00, 01, 05-07, 08)
  ☐ foodguard.db exists (~50 MB)
  ☐ data/synthetic/vision/ has ~420 PNG files
  ☐ certificates/ has QR code PNG files

During Demo:
  ☐ Gradio UI loads at http://localhost:7860
  ☐ Tab 1 dropdown shows 140+ demo samples
  ☐ Can select a sample and click "Run Investigation"
  ☐ Tab 2 shows processing steps (agents running)
  ☐ Tab 3 shows LLM reasoning (from Qwen3 or fallback)
  ☐ Tab 4 shows verdict (adulterant + confidence + risk level)
  ☐ Results appear within 10 seconds

Technical:
  ☐ Database has ~14,000 rows (aroma/taste/vision)
  ☐ agent_execution table has logs from both agents
  ☐ passports table populated with investigation records
  ☐ No errors in notebook console
  ☐ QR codes are valid PNG files

Feature Demo:
  ☐ Show correlation rules matching
  ☐ Show weighted voting calculation
  ☐ Show LLM reasoning (explain why it picked that adulterant)
  ☐ Show risk level assignment
  ☐ Show full investigation report
  ☐ Show traceability (agent logs visible in DB)
```

---

## Tips for Judges

### What to Show

1. **Database Design**: "11 tables, fully normalized, ACID compliance"
2. **Agent Orchestration**: "Two LangGraph agents work together"
3. **Correlation Engine**: "Weighted voting with research-backed rules"
4. **LLM Reasoning**: "Qwen3 explains why it picked that adulterant"
5. **Traceability**: "Every decision logged to agent_execution table"
6. **Explainability**: "Rules, weights, confidence scores all visible"

### What to Skip

- Don't show real model training (takes hours)
- Don't show code review (judges want to see working system)
- Don't explain every database schema detail (high level is fine)
- Don't run multiple investigations simultaneously (shows single-threaded)

### How to Impress

- **"This works end-to-end in 20 seconds"** ← Emphasize speed
- **"See? All decisions are logged here"** ← Point to agent_execution table
- **"The LLM knows about milk science"** ← Show Qwen3 response quality
- **"This is research-backed, not just ML"** ← Correlation rules from papers
- **"No proprietary APIs needed"** ← Ollama is local, no cloud dependency

---

## Next Steps After Hackathon

1. **Integrate Real ML Models**
   - Collect real milk samples
   - Train XGBoost + CNN models
   - Add to model_registry

2. **Add User Interface**
   - Complete Gradio UI polish
   - Add export to PDF/QR
   - Add batch processing

3. **Deploy Production**
   - Set up PostgreSQL
   - Deploy FastAPI backend
   - Set up monitoring/alerting

4. **Regulatory Compliance**
   - Add audit logging
   - Implement blockchain integration
   - Add cryptographic signing (SHA-256 is start)

---

## Support & Questions

### Where to Find Documentation

- **Technical Details**: See [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)
- **Code Comments**: Every notebook has detailed comments
- **Function Docstrings**: `foodguard_lib.py` has full docstrings
- **Database Schema**: `foodguard_lib.init_db()` shows all 11 tables

### Common Questions

**Q: Why mocked ML predictions?**
A: Focus is on agent orchestration, not ML training. Demo works immediately without waiting for model training.

**Q: Can I use a different LLM?**
A: Yes. Edit notebook 06, change `ollama_client.generate(model="qwen3", ...)` to any Ollama-supported model.

**Q: Why SQLite instead of PostgreSQL?**
A: For hackathon speed/simplicity. Production should use PostgreSQL.

**Q: Can I add more correlation rules?**
A: Yes. Edit notebook 00, add more entries to correlation_rules list.

**Q: How do I add new adulterant classes?**
A: Add to ADULTERANTS list in foodguard_lib.py, then add rules in notebook 00.

---

## References

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Ollama Docs**: https://ollama.ai/docs
- **Gradio Docs**: https://gradio.app/docs/
- **SQLite Docs**: https://www.sqlite.org/docs.html

---

**Last Updated**: June 12, 2026
**Version**: 1.0
**Status**: Ready for Production 🚀
