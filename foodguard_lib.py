"""
FoodGuard AI — Shared Utilities Library
Contains: DB helpers, ID generators, LLM caller, schema initialization
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import uuid
import requests
import hashlib
import qrcode
import io
from contextlib import contextmanager
from pathlib import Path

# ============= Constants =============

DB_PATH = "foodguard.db"

ADULTERANTS = [
    "Fresh",
    "Water",
    "Urea",
    "Detergent",
    "Formalin",
    "H2O2",
    "Spoiled"
]

# ============= DB Helpers =============

@contextmanager
def get_db_connection(db_path: str = DB_PATH):
    """Context manager for SQLite connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def execute_query(db_path: str, query: str, params: tuple = ()) -> List[sqlite3.Row]:
    """Execute SELECT query, return results as list of Row objects."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

def execute_insert(db_path: str, query: str, params: tuple = ()) -> int:
    """Execute INSERT/UPDATE query, return lastrowid."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.lastrowid

def dict_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert sqlite3.Row to dict."""
    if row is None:
        return {}
    return dict(row)

# ============= ID Generators =============

def generate_id(prefix: str) -> str:
    """Generate unique IDs: ARO-xxxx, TAS-xxxx, VIS-xxxx, INV-xxxx, etc."""
    suffix = str(uuid.uuid4())[:8].upper()
    return f"{prefix}-{suffix}"

def generate_batch_id() -> str:
    """Generate batch ID for a sample investigation."""
    return generate_id("BATCH")

def generate_investigation_id() -> str:
    """Generate investigation ID."""
    return generate_id("INV")

def generate_aroma_id() -> str:
    """Generate aroma analysis ID."""
    return generate_id("ARO")

def generate_taste_id() -> str:
    """Generate taste analysis ID."""
    return generate_id("TAS")

def generate_vision_id() -> str:
    """Generate vision analysis ID."""
    return generate_id("VIS")

def generate_correlation_id() -> str:
    """Generate correlation ID."""
    return generate_id("CORR")

def generate_passport_id() -> str:
    """Generate passport ID."""
    return generate_id("PASS")

# ============= LLM Helpers =============

def get_ollama_client(base_url: str = "http://localhost:11434"):
    """Return Ollama client."""
    return OllamaClient(base_url=base_url)

class OllamaClient:
    """Wrapper for Ollama API calls."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def generate(self, model: str, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """
        Call Ollama generate endpoint.
        Returns: {"response": "...", "context": [...], "total_duration": ..., ...}
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": stream},
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Fallback: return mock response
            print(f"[LLM Error] {e}. Returning fallback response.")
            return {
                "response": f"[MOCK LLM] Unable to reach Ollama. Would reason about {model} with prompt.",
                "fallback": True
            }

# ============= Hash & QR Helpers =============

def compute_sha256(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of JSON dict."""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()

def generate_qr_code(data: str, output_path: Optional[str] = None) -> str:
    """
    Generate QR code from data string.
    If output_path provided, save to file and return path.
    Otherwise return as base64 or image object.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path

    return img

# ============= Schema Initialization =============

def init_db(db_path: str = DB_PATH):
    """Create all tables if not exist."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        statements = [
            """CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                predicted_class TEXT,
                confidence REAL,
                risk_level TEXT,
                reasoning TEXT,
                model_version TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",

            """CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                evidence_type TEXT,
                raw_data TEXT,
                prediction TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            )""",

            """CREATE TABLE IF NOT EXISTS aroma_analysis (
                id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                ammonia REAL, alcohol REAL, voc REAL, sulfur REAL, hydrocarbon REAL,
                predicted_class TEXT,
                confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",

            """CREATE TABLE IF NOT EXISTS taste_analysis (
                id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                sweetness REAL, saltiness REAL, sourness REAL, bitterness REAL,
                umami REAL, astringency REAL,
                predicted_class TEXT,
                confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",

            """CREATE TABLE IF NOT EXISTS vision_analysis (
                id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                image_path TEXT,
                class_counts_json TEXT NOT NULL,
                total_objects INT NOT NULL DEFAULT 0,
                unique_classes TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",

            """CREATE TABLE IF NOT EXISTS correlation_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT,
                pattern_json TEXT,
                target_adulterant TEXT,
                confidence_boost REAL,
                explanation TEXT,
                weight REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",

            """CREATE TABLE IF NOT EXISTS correlations (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                matched_rule_ids TEXT,
                candidate_adulterants TEXT,
                final_adulterant TEXT,
                final_confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            )""",

            """CREATE TABLE IF NOT EXISTS agent_execution (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                agent_name TEXT,
                input_data TEXT,
                output_data TEXT,
                reasoning TEXT,
                execution_time_ms REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            )""",

            """CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                report_json TEXT,
                report_markdown TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            )""",

            """CREATE TABLE IF NOT EXISTS passports (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                passport_json TEXT,
                hash_sha256 TEXT,
                qr_code_path TEXT,
                blockchain_status TEXT DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            )""",

            """CREATE TABLE IF NOT EXISTS model_registry (
                id TEXT PRIMARY KEY,
                model_name TEXT,
                model_type TEXT,
                accuracy REAL,
                model_version TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",
        ]

        for stmt in statements:
            cursor.execute(stmt)

        conn.commit()
        print(f"[DB] Schema initialized: {db_path}")

# ============= Data Insertion Helpers =============

def insert_aroma_analysis(
    batch_id: str,
    ammonia: float, alcohol: float, voc: float, sulfur: float, hydrocarbon: float,
    predicted_class: str, confidence: float,
    db_path: str = DB_PATH
) -> str:
    """Insert aroma analysis record, return ID."""
    aroma_id = generate_aroma_id()
    query = """INSERT INTO aroma_analysis
        (id, batch_id, ammonia, alcohol, voc, sulfur, hydrocarbon, predicted_class, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    execute_insert(db_path, query, (
        aroma_id, batch_id, ammonia, alcohol, voc, sulfur, hydrocarbon, predicted_class, confidence
    ))
    return aroma_id

def insert_taste_analysis(
    batch_id: str,
    sweetness: float, saltiness: float, sourness: float, bitterness: float,
    umami: float, astringency: float,
    predicted_class: str, confidence: float,
    db_path: str = DB_PATH
) -> str:
    """Insert taste analysis record, return ID."""
    taste_id = generate_taste_id()
    query = """INSERT INTO taste_analysis
        (id, batch_id, sweetness, saltiness, sourness, bitterness, umami, astringency, predicted_class, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    execute_insert(db_path, query, (
        taste_id, batch_id, sweetness, saltiness, sourness, bitterness, umami, astringency, predicted_class, confidence
    ))
    return taste_id

def insert_vision_analysis(
    batch_id: str,
    image_path: str, deposit_type: str,
    predicted_class: str, confidence: float,
    db_path: str = DB_PATH
) -> str:
    """Insert vision analysis record, return ID."""
    vision_id = generate_vision_id()
    query = """INSERT INTO vision_analysis
        (id, batch_id, image_path, deposit_type, predicted_class, confidence)
        VALUES (?, ?, ?, ?, ?, ?)"""
    execute_insert(db_path, query, (
        vision_id, batch_id, image_path, deposit_type, predicted_class, confidence
    ))
    return vision_id

def insert_investigation(
    batch_id: str,
    predicted_class: str, confidence: float,
    db_path: str = DB_PATH
) -> str:
    """Insert investigation record, return ID."""
    investigation_id = generate_investigation_id()
    query = """INSERT INTO investigations
        (id, batch_id, predicted_class, confidence)
        VALUES (?, ?, ?, ?)"""
    execute_insert(db_path, query, (investigation_id, batch_id, predicted_class, confidence))
    return investigation_id

def insert_correlation_rule(
    rule_name: str, pattern_json: Dict, target_adulterant: str,
    confidence_boost: float, explanation: str, weight: float,
    db_path: str = DB_PATH
) -> int:
    """Insert correlation rule, return ID."""
    query = """INSERT INTO correlation_rules
        (rule_name, pattern_json, target_adulterant, confidence_boost, explanation, weight)
        VALUES (?, ?, ?, ?, ?, ?)"""
    rule_id = execute_insert(db_path, query, (
        rule_name, json.dumps(pattern_json), target_adulterant, confidence_boost, explanation, weight
    ))
    return rule_id

def insert_agent_execution(
    investigation_id: str, agent_name: str,
    input_data: Dict, output_data: Dict, reasoning: str = "",
    execution_time_ms: float = 0.0,
    db_path: str = DB_PATH
) -> str:
    """Log agent execution, return ID."""
    exec_id = generate_id("EXEC")
    query = """INSERT INTO agent_execution
        (id, investigation_id, agent_name, input_data, output_data, reasoning, execution_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)"""
    execute_insert(db_path, query, (
        exec_id, investigation_id, agent_name,
        json.dumps(input_data), json.dumps(output_data), reasoning, execution_time_ms
    ))
    return exec_id

# ============= Data Query Helpers =============

def get_aroma_analysis(batch_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    """Fetch latest aroma analysis for batch."""
    query = "SELECT * FROM aroma_analysis WHERE batch_id = ? ORDER BY created_at DESC LIMIT 1"
    rows = execute_query(db_path, query, (batch_id,))
    return dict_from_row(rows[0]) if rows else None

def get_taste_analysis(batch_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    """Fetch latest taste analysis for batch."""
    query = "SELECT * FROM taste_analysis WHERE batch_id = ? ORDER BY created_at DESC LIMIT 1"
    rows = execute_query(db_path, query, (batch_id,))
    return dict_from_row(rows[0]) if rows else None

def get_vision_analysis(batch_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    """Fetch latest vision analysis for batch."""
    query = "SELECT * FROM vision_analysis WHERE batch_id = ? ORDER BY created_at DESC LIMIT 1"
    rows = execute_query(db_path, query, (batch_id,))
    return dict_from_row(rows[0]) if rows else None

def get_correlation_rules(db_path: str = DB_PATH) -> List[Dict]:
    """Fetch all correlation rules."""
    query = "SELECT * FROM correlation_rules ORDER BY weight DESC"
    rows = execute_query(db_path, query)
    return [dict_from_row(row) for row in rows]

def get_investigation(investigation_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    """Fetch investigation by ID."""
    query = "SELECT * FROM investigations WHERE id = ?"
    rows = execute_query(db_path, query, (investigation_id,))
    return dict_from_row(rows[0]) if rows else None

def update_investigation(investigation_id: str, updates: Dict, db_path: str = DB_PATH):
    """Update investigation record."""
    set_clauses = ", ".join([f"{k} = ?" for k in updates.keys()])
    query = f"UPDATE investigations SET {set_clauses} WHERE id = ?"
    params = tuple(updates.values()) + (investigation_id,)
    execute_insert(db_path, query, params)

# ============= Report Generation =============

def generate_report_markdown(
    investigation_id: str,
    suspected_adulterant: str,
    confidence: float,
    risk_level: str,
    reasoning: str,
    aroma_conf: float = 0.0,
    taste_conf: float = 0.0,
    vision_conf: float = 0.0,
    matched_rules: List[str] = None
) -> str:
    """Generate markdown report."""

    if matched_rules is None:
        matched_rules = []

    md = f"""# Food Guard AI — Investigation Report

## Investigation ID
`{investigation_id}`

## Verdict
- **Suspected Adulterant**: {suspected_adulterant}
- **Overall Confidence**: {confidence:.1%}
- **Risk Level**: **{risk_level}**

## Evidence Summary
| Modality | Confidence |
|----------|-----------|
| Aroma (E-Nose) | {aroma_conf:.1%} |
| Taste (E-Tongue) | {taste_conf:.1%} |
| Vision (Deposit Pattern) | {vision_conf:.1%} |

## Matched Correlation Rules
{len(matched_rules)} rule(s) matched:
"""

    for i, rule in enumerate(matched_rules, 1):
        md += f"\n{i}. {rule}"

    md += f"""

## AI Reasoning
{reasoning}

---
*Generated by FoodGuard AI — Hackathon Edition*
"""
    return md

# ============= Utility Helpers =============

def ensure_directories():
    """Create necessary directories if they don't exist."""
    Path("data/synthetic/vision").mkdir(parents=True, exist_ok=True)
    Path("certificates").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("notebooks").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Test: initialize DB
    init_db()
    ensure_directories()
    print("[OK] Library initialized successfully")
