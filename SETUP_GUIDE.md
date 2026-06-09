# 🚀 FoodGuard AI — Docker Setup & Quick Start

---

## 🐳 Docker Compose Setup

### Start the Application

```bash
cd /projects/food-guard-ai

# Build and start containers
docker compose up --build

# In another terminal, check status
docker ps
```

**Expected Output:**
```
CONTAINER ID   IMAGE                    PORTS                    STATUS
abc123def456   foodguard-app            0.0.0.0:7860->7860/tcp   healthy
```

### Access the UI

Open browser: **http://localhost:7860**

**5 Tabs:**
1. **📋 Submit Sample** — Batch name + image → run investigation
2. **📊 Investigation Dashboard** — Multi-modal results
3. **🔐 Food Passport** — Certificate + hash + QR
4. **✅ Verify Certificate** — Hash verification
5. **⚙️ System Info** — Correlation rules status

### Stop Application

```bash
# Graceful stop
docker compose down

# Stop and remove volumes (WARNING: deletes database)
docker compose down -v
```

---

## 🔧 Docker Compose Infrastructure

### Service: `foodguard-app` (Single Service)
- **Role:** Main Gradio application + LangGraph orchestrator
- **Image:** Built from `Dockerfile`
- **Port:** `7860` (Gradio)
- **Volumes:**
  - `/data` → `foodguard_data` (SQLite persistence)
  - `./certificates/` → `/app/certificates/` (passport + QR output)
  - `./data/` → `/app/data/` (synthetic datasets, read-only)
- **Health Check:** Every 30s via `curl http://localhost:7860`
- **Auto-restart:** Unless stopped

> **Note:** SQLite persistence is handled via Docker volume `foodguard_data`. No separate monitoring service is needed — data persists automatically across restarts.

### Volume: `foodguard_data`
- **Type:** Local (auto-managed)
- **Purpose:** Persists SQLite database across container restarts
- **Location:** Docker internal storage

### Network: `foodguard-network`
- **Type:** Bridge
- **Purpose:** Internal service-to-service communication (reserved for future multi-service expansion)

---

## 🚀 Quick Start Examples

### Example 1: Run with Defaults
```bash
docker compose up --build
# App available at http://localhost:7860
```

### Example 2: Run in Background
```bash
docker compose up -d --build
# Returns container IDs

# View logs
docker compose logs -f foodguard-app

# Stop background
docker compose stop
```

### Example 3: Custom Port
Edit `docker-compose.yml`:
```yaml
services:
  foodguard-app:
    ports:
      - "8080:7860"  # Use http://localhost:8080
```

Then:
```bash
docker compose up --build
```

### Example 4: Persist Database Externally
Edit `docker-compose.yml`:
```yaml
volumes:
  foodguard_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /projects/food-guard-ai  # Local path
```

---

## 📝 Environment Variables

**Inside Container:**
```bash
FOODGUARD_DB_PATH=/data/foodguard.db       # SQLite location
PYTHONUNBUFFERED=1                         # Unbuffered output
```

**For Phase 5 (LLM integration):**
```bash
OPENAI_API_KEY=sk-xxx...                   # GPT-4 for Food Safety Agent
# Add to docker-compose.yml environment or .env
```

---

## 📊 Database

### Schema (8 SQLModel Tables)
1. **batches** — Investigation metadata
2. **vision_analysis** — YOLOv12 results (score, label, findings)
3. **aroma_analysis** — E-Nose XGBoost results
4. **taste_analysis** — E-Tongue XGBoost results
5. **correlation_rules** — Rule base (80+ rules)
6. **correlations** — Multi-modal matching (pattern_type, confidence_delta)
7. **investigations** — Food safety verdicts (overall_score, risk_level)
8. **passports** — Certificates (hash, qr_code, blockchain_status)

### Access Database

```bash
# Inside container
docker exec foodguard-app sqlite3 /data/foodguard.db ".tables"

# Or mount volume locally
docker run -it -v foodguard_data:/data alpine:latest sh
sqlite3 /data/foodguard.db "SELECT COUNT(*) FROM batches;"
```

---

## 🧪 Testing the Pipeline

### Test 1: Basic Investigation
```bash
# In Gradio UI
- Tab 1: Submit Sample
  - Batch Name: "Test-001"
  - Image: (any jpg/png)
  - Click "Run Investigation"

# Observe:
- Tab 2: Investigation results
- Tab 3: Passport with certificate hash
- Tab 5: System info shows correlation rules loaded
```

### Test 2: Database Inspection
```bash
docker exec foodguard-app sqlite3 /data/foodguard.db "SELECT * FROM batches LIMIT 5;"
docker exec foodguard-app sqlite3 /data/foodguard.db "SELECT COUNT(*) FROM correlation_rules;"
```

### Test 3: Logs & Debugging
```bash
# View real-time logs
docker compose logs -f foodguard-app

# View specific error
docker compose logs foodguard-app | grep ERROR

# Check health
docker compose ps
```

---

## 🔍 Troubleshooting

### ❌ Port 7860 Already in Use
```bash
# Find process
lsof -i :7860

# Kill it
kill -9 <PID>

# Or use different port in docker-compose.yml
```

### ❌ Database Locked
```bash
# Remove containers + volumes
docker compose down -v

# Rebuild
docker compose up --build
```

### ❌ Build Fails (Missing Dependencies)
```bash
# Update requirements.txt
pip install -r requirements.txt --dry-run  # Test locally

# Then rebuild
docker compose build --no-cache foodguard-app
```

### ❌ Gradio Not Responsive
```bash
# Check logs
docker compose logs foodguard-app

# Restart
docker compose restart foodguard-app

# Force rebuild
docker compose down
docker compose up --build --force-recreate
```

### ❌ Volume Data Lost
```bash
# Check volume
docker volume ls | grep foodguard

# Inspect
docker volume inspect foodguard_data

# Backup before cleanup
docker run -v foodguard_data:/data -v $(pwd):/backup alpine cp /data/foodguard.db /backup/
```

---

## 📈 Development Workflow

### Phase 2: Add Synthetic Datasets
```bash
# Generate datasets into data/synthetic/
python scripts/generate_enose.py     # → enose_dataset.csv
python scripts/generate_etongue.py   # → etongue_dataset.csv

# Restart container (auto-loaded)
docker compose restart foodguard-app
```

### Phase 3: Train Models
```bash
# Inside container
docker exec -it foodguard-app bash
python ml/train_yolo.py
python ml/train_enose.py
python ml/train_etongue.py

# Models saved to ml/models/
```

### Phase 4+: Update Agents
```bash
# Edit agents/vision.py, aroma.py, etc.
# Rebuild container
docker compose up --build

# Changes auto-reload (no container restart needed if just app.py changes)
```

---

## 📦 Deployment Checklist

- [ ] All agents implemented (supervisor → passport)
- [ ] Models trained (YOLOv12, XGBoost)
- [ ] Correlation rules loaded (80+ rules)
- [ ] Database schema verified (8 tables)
- [ ] Gradio UI tested (all 5 tabs working)
- [ ] Docker builds without errors
- [ ] docker compose up runs without issues
- [ ] Certificates generated with hash + QR
- [ ] Blockchain stub ready for integration

---

## 🎯 What's Ready Now (Phase 1 ✅)

✅ **Modular folder structure** — 7 directories (agents, models, ml, utils, data, certificates, documentation)
✅ **8 agent files** — LangGraph DAG nodes (7 stubs with Phase 3–6 TODOs, 1 fully implemented: `agents/passport.py`)
✅ **Database schema** — 8 SQLModel tables with proper relationships (Batch, Vision, Aroma, Taste, Correlation, Correlations, Investigations, Passports)
✅ **Gradio UI** — 5 tabs fully wired to LangGraph orchestrator (Submit Sample, Dashboard, Passport, Verify, System Info)
✅ **Docker infrastructure** — Simplified single-service setup (Dockerfile + docker-compose.yml with auto-managed volume persistence)
✅ **Synthetic data placeholders** — correlation_rules.csv (15 rules, expandable to 80+), enose_dataset.csv, etongue_dataset.csv
✅ **ID generation utility** — Sequential format with date prefix (batch_id, vision_id, etc.)
✅ **5 comprehensive guides** — PROJECT_PLAN.md, README_DOCKER.md, DEPLOYMENT.md, SETUP_GUIDE.md, COMPLETION_SUMMARY.md
✅ **Tested LangGraph routing** — Supervisor → Parallel (Vision/Aroma/Taste) → Correlation → Food Safety → Reviewer → Passport → END

---

## 📋 Next: Phase 2 (Day 2)

- [ ] Generate synthetic vision images (200–300/class × 5 classes)
- [ ] Generate E-Nose CSV (500/class × 6 adulterants)
- [ ] Generate E-Tongue CSV (500/class × 6 adulterants)
- [ ] Seed correlation_rules table (80–100 rules)
- [ ] Test data loading in Docker

---

## 🆘 Support & Questions

**Logs:**
```bash
docker compose logs foodguard-app
```

**Database:**
```bash
docker exec foodguard-app sqlite3 /data/foodguard.db ".schema"
```

**Test connection:**
```bash
docker run --network foodguard-network -it alpine:latest ping foodguard-app
```

---

**Status:** Phase 1 Complete ✅ (Docker simplified, ready for Phase 2 data generation)

