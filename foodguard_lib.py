"""
FoodGuard AI — Shared Utilities Library
Contains: DB helpers, ID generators, LLM caller, schema initialization
"""

# ============================================================================
# ALL IMPORTS
# ============================================================================
import hashlib
import io
import json
import os
import random
import shutil
import sqlite3
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import qrcode
import requests
import torch
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Suppress runtime noise from third-party libraries
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# ============= Constants =============

DB_PATH = "foodguard.db"

# ============================================================================
# CENTRALIZED PATH & FRAMEWORK CONSTANTS
# ============================================================================
ROOT_PATH = "/workspace/shared/food-guard-ai"
VAL_FOLDER = f"{ROOT_PATH}/data/synthetic/vision"
YAML_CONFIG_PATH = f"{ROOT_PATH}/ml/milk.yaml"

# Distinct paths to prevent model collision during training
YOLO_CLS_WEIGHTS = f"{ROOT_PATH}/ml/milk_cls.pt"
YOLO_DET_WEIGHTS = f"{ROOT_PATH}/ml/milk_det.pt"

INPUT_VAL_FOLDER = f"{ROOT_PATH}/data/val/"
OUTPUT_DIR = f"{ROOT_PATH}/output"
JSON_OUTPUT_FOLDER = f"{OUTPUT_DIR}/data/yolo_predictions"

ADULTERANTS = [
    "Fresh",
    "Water",
    "Urea",
    "Detergent",
    "Formalin",
    "H2O2",
    "Spoiled"
]

# --- Shared Framework Runtime Output Trajectories ---
YOLO_DET_PROJECT_NAME = "milk_detection_framework"
YOLO_DET_RUN_NAME = "yolo_milk_det_run"

YOLO_CLS_PROJECT_NAME = "milk_classification_framework"
YOLO_CLS_RUN_NAME = "yolo_milk_cls_run"

CRM_PATH = f"{ROOT_PATH}/data/certified_reference_materials(crm).json"
PROMPT_PATH = f"{ROOT_PATH}/data/foodguard_crm_agent_prompt.py"
ENRICHED_SAMPLES_PATH = f"{OUTPUT_DIR}/enriched_samples.json"
BATCH_RISK_SUMMARY_PATH = f"{OUTPUT_DIR}/batch_risk_summary.json"
FOOD_SAFETY_REPORT_JSON = f"{OUTPUT_DIR}/food_safety_report.json"
FOOD_SAFETY_REPORT_MD = f"{OUTPUT_DIR}/food_safety_report.md"
SHAP_GLOBAL_IMPORTANCE_PATH = f"{ROOT_PATH}/ml/shap_global_importance.png"
BASE_PATH = Path(ROOT_PATH)

BASE_URL_MODEL = "http://localhost:8000/v1"
OPENAI_API_KEY = "abc-123"
LLM = "Qwen3-4B"

# ============= Batch Data =============
SOURCES = [
    "Daily Cooperative Center",
    "Village Milk Collection Unit",
    "Automated Milk Testing Lab",
    "Organic Farm Network"
]

DESCRIPTIONS = [
    "High quality A2 milk verified for purity",
    "Adulteration passed with no contamination",
    "Antibiotic milk confirmed by lab test"
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
                batch_id TEXT,
                status TEXT DEFAULT 'in_progress',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                FOREIGN KEY (batch_id) REFERENCES batches(id)
            )""",

            """CREATE TABLE IF NOT EXISTS batches (
                id TEXT PRIMARY KEY,
                source TEXT,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",

            """CREATE TABLE IF NOT EXISTS aroma_analysis (
                sample_id TEXT PRIMARY KEY,
                batch_id TEXT,
                ground_truth TEXT,
                ammonia REAL,
                alcohol REAL,
                voc REAL,
                sulfur REAL,
                hydrocarbon REAL,
                predicted_class TEXT,
                confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (batch_id) REFERENCES batches(id)
            )""",
            """CREATE TABLE IF NOT EXISTS taste_analysis (
                sample_id TEXT PRIMARY KEY,
                batch_id TEXT,
                ground_truth TEXT,
                sweetness REAL,
                saltiness REAL,
                sourness REAL,
                bitterness REAL,
                umami REAL,
                astringency REAL,
                predicted_class TEXT,
                confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (batch_id) REFERENCES batches(id)
            )""",
            """CREATE TABLE IF NOT EXISTS vision_analysis (
                sample_id TEXT PRIMARY KEY,
                batch_id TEXT,
                ground_truth TEXT DEFAULT NULL,
                image_path TEXT,
                deposit_type TEXT,
                predicted_class TEXT,
                confidence TEXT,
                object_detection_info TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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

def insert_investigation(
    batch_id: str = None,
    status: str = "in_progress",
    db_path: str = DB_PATH
) -> str:
    investigation_id = generate_investigation_id()
    query = """INSERT INTO investigations (id, batch_id, status) VALUES (?, ?, ?)"""
    execute_insert(db_path, query, (investigation_id, batch_id, status))
    return investigation_id

def insert_batch(
    source: str = None,
    description: str = None,
    db_path: str = DB_PATH
) -> str:
    batch_id = generate_batch_id()
    query = """INSERT INTO batches (id, source, description) VALUES (?, ?, ?)"""
    execute_insert(db_path, query, (batch_id, source, description))
    return batch_id

def populate_initial_batches(db_path: str = DB_PATH) -> List[str]:
    """
    Populate batches table with initial SOURCES and DESCRIPTIONS.
    Creates combinations of sources and descriptions.
    Returns list of created batch IDs.
    """
    batch_ids = []

    # Create combinations of sources and descriptions
    for source in SOURCES:
        for description in DESCRIPTIONS:
            batch_id = insert_batch(source=source, description=description, db_path=db_path)
            batch_ids.append(batch_id)

    print(f"[DB] Created {len(batch_ids)} initial batches")
    return batch_ids

def insert_aroma_analysis(
    batch_id: str,
    ammonia: float, alcohol: float, voc: float, sulfur: float, hydrocarbon: float,
    predicted_class: str, confidence: float,
    ground_truth: str = None,
    db_path: str = DB_PATH
) -> str:
    query = """INSERT INTO aroma_analysis
        (sample_id, batch_id, ground_truth, ammonia, alcohol, voc, sulfur, hydrocarbon, predicted_class, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    execute_insert(db_path, query, (
        batch_id, ground_truth, ammonia, alcohol, voc, sulfur, hydrocarbon, predicted_class, confidence
    ))
    return batch_id

def insert_taste_analysis(
    batch_id: str,
    sweetness: float, saltiness: float, sourness: float, bitterness: float,
    umami: float, astringency: float,
    predicted_class: str, confidence: float,
    ground_truth: str = None,
    db_path: str = DB_PATH
) -> str:
    query = """INSERT INTO taste_analysis
        (sample_id, batch_id, ground_truth, sweetness, saltiness, sourness, bitterness, umami, astringency, predicted_class, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    execute_insert(db_path, query, (
        batch_id, ground_truth, sweetness, saltiness, sourness, bitterness, umami, astringency, predicted_class, confidence
    ))
    return batch_id


# ===========================================================================
# Generate Data for e-nose and e-tougue
# ===========================================================================
def generate_enose_data(n_samples_per_class=200):
    """
    Generate E-Nose synthetic data.
    Returns DataFrame with columns: adulterant, ammonia, alcohol, voc, sulfur, hydrocarbon
    """

    # Sensor ranges and shifts per adulterant (from literature)
    enose_profiles = {
        "Fresh": {
            "ammonia": (0.1, 0.5),
            "alcohol": (0.2, 0.8),
            "voc": (0.3, 0.9),
            "sulfur": (0.05, 0.3),
            "hydrocarbon": (0.1, 0.6)
        },
        "Water": {  # Diluted signals
            "ammonia": (0.05, 0.25),
            "alcohol": (0.1, 0.4),
            "voc": (0.1, 0.5),
            "sulfur": (0.02, 0.15),
            "hydrocarbon": (0.05, 0.3)
        },
        "Urea": {  # High ammonia
            "ammonia": (1.5, 3.0),
            "alcohol": (0.2, 0.8),
            "voc": (0.3, 0.9),
            "sulfur": (0.1, 0.5),
            "hydrocarbon": (0.1, 0.6)
        },
        "Detergent": {  # Variable profile, some alcohol
            "ammonia": (0.2, 0.6),
            "alcohol": (0.5, 1.5),
            "voc": (0.8, 2.0),
            "sulfur": (0.2, 0.8),
            "hydrocarbon": (0.2, 1.0)
        },
        "Formalin": {  # Pungent signals
            "ammonia": (0.5, 1.5),
            "alcohol": (1.5, 3.5),
            "voc": (2.0, 4.5),
            "sulfur": (0.5, 1.5),
            "hydrocarbon": (0.3, 1.0)
        },
        "H2O2": {  # Oxidative markers
            "ammonia": (0.1, 0.4),
            "alcohol": (0.3, 1.0),
            "voc": (0.5, 1.5),
            "sulfur": (0.05, 0.25),
            "hydrocarbon": (0.05, 0.3)
        },
        "Spoiled": {  # Acidic fermentation markers
            "ammonia": (0.3, 1.0),
            "alcohol": (0.5, 2.0),
            "voc": (0.8, 2.5),
            "sulfur": (0.3, 1.0),
            "hydrocarbon": (0.1, 0.8)
        }
    }

    data = []
    for adulterant in enose_profiles.keys():
        profile = enose_profiles[adulterant]
        for _ in range(n_samples_per_class):
            sample = {"adulterant": adulterant}
            for sensor, (min_val, max_val) in profile.items():
                # Add Gaussian noise around the range
                base = np.random.uniform(min_val, max_val)
                noise = np.random.normal(0, 0.1 * (max_val - min_val))
                sample[sensor] = max(0, base + noise)
            data.append(sample)

    return pd.DataFrame(data)


def generate_etongue_data(n_samples_per_class=200):
    """
    Generate E-Tongue synthetic data.
    Returns DataFrame with columns: adulterant, sweetness, saltiness, sourness, bitterness, umami, astringency
    """

    etongue_profiles = {
        "Fresh": {
            "sweetness": (4.0, 6.0),
            "saltiness": (1.0, 2.5),
            "sourness": (0.5, 1.5),
            "bitterness": (0.1, 0.5),
            "umami": (0.5, 1.5),
            "astringency": (0.2, 0.8)
        },
        "Water": {  # Diluted taste
            "sweetness": (2.0, 3.5),
            "saltiness": (0.3, 1.0),
            "sourness": (0.2, 0.8),
            "bitterness": (0.05, 0.2),
            "umami": (0.1, 0.5),
            "astringency": (0.05, 0.3)
        },
        "Urea": {  # High bitterness
            "sweetness": (2.0, 4.0),
            "saltiness": (1.5, 3.0),
            "sourness": (0.5, 1.5),
            "bitterness": (2.0, 4.0),  # ** MARKER
            "umami": (0.2, 1.0),
            "astringency": (1.0, 2.0)  # ** MARKER
        },
        "Detergent": {  # Harsh, bitter, soapy
            "sweetness": (0.5, 1.5),
            "saltiness": (2.0, 3.5),
            "sourness": (0.3, 1.0),
            "bitterness": (2.5, 4.5),  # ** MARKER
            "umami": (0.1, 0.5),
            "astringency": (2.0, 3.5)  # ** MARKER
        },
        "Formalin": {  # Pungent, sour, harsh
            "sweetness": (0.2, 1.0),
            "saltiness": (1.0, 2.5),
            "sourness": (3.0, 5.0),  # ** MARKER
            "bitterness": (1.5, 3.0),
            "umami": (0.1, 0.5),
            "astringency": (3.0, 4.5)  # ** MARKER
        },
        "H2O2": {  # Clean, faint
            "sweetness": (3.0, 5.0),
            "saltiness": (0.2, 1.0),
            "sourness": (0.1, 0.5),
            "bitterness": (0.05, 0.3),
            "umami": (0.1, 0.5),
            "astringency": (0.1, 0.5)
        },
        "Spoiled": {  # Sour, acidic
            "sweetness": (1.0, 2.5),
            "saltiness": (1.5, 2.5),
            "sourness": (3.5, 5.5),  # ** MARKER
            "bitterness": (0.5, 1.5),
            "umami": (0.1, 0.5),
            "astringency": (1.5, 3.0)  # ** MARKER
        }
    }

    data = []
    for adulterant in etongue_profiles.keys():
        profile = etongue_profiles[adulterant]
        for _ in range(n_samples_per_class):
            sample = {"adulterant": adulterant}
            for sensor, (min_val, max_val) in profile.items():
                base = np.random.uniform(min_val, max_val)
                noise = np.random.normal(0, 0.15 * (max_val - min_val))
                sample[sensor] = max(0, base + noise)
            data.append(sample)

    return pd.DataFrame(data)



# ----------------------------------------------------------------------------
# CHANGE: Completely refactored this function parameters and query signature
# to directly write columns mapped from the nested evaluation payload.
# ----------------------------------------------------------------------------
def insert_vision_analysis(
    sample_id: str,
    batch_id: str,
    image_path: str,
    deposit_type: str,
    predicted_class: str,
    confidence: str,
    object_detection_info: str,
    ground_truth: str = None,
    db_path: str = DB_PATH
) -> str:
    query = """INSERT INTO vision_analysis
        (sample_id, batch_id, ground_truth, image_path, deposit_type, predicted_class, confidence, object_detection_info)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
    execute_insert(db_path, query, (
        sample_id, batch_id, ground_truth, image_path, deposit_type, predicted_class, confidence, object_detection_info
    ))
    return sample_id

def insert_agent_execution(
    investigation_id: str, agent_name: str,
    input_data: Dict, output_data: Dict, reasoning: str = "",
    execution_time_ms: float = 0.0,
    db_path: str = DB_PATH
) -> str:
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
    query = "SELECT * FROM aroma_analysis WHERE batch_id = ? ORDER BY created_at DESC LIMIT 1"
    rows = execute_query(db_path, query, (batch_id,))
    return dict_from_row(rows[0]) if rows else None

def get_taste_analysis(batch_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    query = "SELECT * FROM taste_analysis WHERE batch_id = ? ORDER BY created_at DESC LIMIT 1"
    rows = execute_query(db_path, query, (batch_id,))
    return dict_from_row(rows[0]) if rows else None

# ----------------------------------------------------------------------------
# CHANGE: Updated function filtering matching key variable from batch_id to sample_id
# ----------------------------------------------------------------------------
def get_vision_analysis(sample_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    query = "SELECT * FROM vision_analysis WHERE sample_id = ? ORDER BY created_at DESC LIMIT 1"
    rows = execute_query(db_path, query, (sample_id,))
    return dict_from_row(rows[0]) if rows else None

def get_correlation_rules(db_path: str = DB_PATH) -> List[Dict]:
    query = "SELECT * FROM correlation_rules ORDER BY weight DESC"
    rows = execute_query(db_path, query)
    return [dict_from_row(row) for row in rows]

def get_investigation(investigation_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    query = "SELECT * FROM investigations WHERE id = ?"
    rows = execute_query(db_path, query, (investigation_id,))
    return dict_from_row(rows[0]) if rows else None

def update_investigation(investigation_id: str, updates: Dict, db_path: str = DB_PATH):
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

def generate_vision_images(n_samples_per_class=20, output_dir="../data/synthetic/vision"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    SIZE = 256
    CENTER = (SIZE // 2, SIZE // 2)
    MAX_RADIUS = SIZE // 2 - 10

    def generate_fresh(img):
        draw = ImageDraw.Draw(img)
        ring_inner = MAX_RADIUS - 30
        ring_outer = MAX_RADIUS
        for r in np.linspace(ring_inner, ring_outer, 5):
            draw.ellipse(
                [CENTER[0]-r, CENTER[1]-r, CENTER[0]+r, CENTER[1]+r],
                outline=(100, 100, 100), width=2
            )

    def generate_water(img):
        draw = ImageDraw.Draw(img)
        r = MAX_RADIUS - 40
        draw.ellipse(
            [CENTER[0]-r, CENTER[1]-r, CENTER[0]+r, CENTER[1]+r],
            outline=(50, 50, 50), width=1
        )

    def generate_urea(img):
        pixels = img.load()
        for _ in range(200):
            x = random.randint(50, SIZE-50)
            y = random.randint(50, SIZE-50)
            dist_from_center = ((x-CENTER[0])**2 + (y-CENTER[1])**2)**0.5
            if dist_from_center < MAX_RADIUS:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if 0 <= x+dx < SIZE and 0 <= y+dy < SIZE:
                            pixels[x+dx, y+dy] = (80, 80, 80)

    def generate_detergent(img):
        draw = ImageDraw.Draw(img)
        for angle in np.linspace(0, 2*np.pi, 12):
            x_end = CENTER[0] + MAX_RADIUS * np.cos(angle)
            y_end = CENTER[1] + MAX_RADIUS * np.sin(angle)
            draw.line([CENTER, (x_end, y_end)], fill=(100, 100, 100), width=2)

    def generate_formalin(img):
        draw = ImageDraw.Draw(img)
        blob_size = MAX_RADIUS // 3
        draw.ellipse(
            [CENTER[0]-blob_size, CENTER[1]-blob_size, CENTER[0]+blob_size, CENTER[1]+blob_size],
            fill=(30, 30, 30)
        )

    def generate_h2o2(img):
        draw = ImageDraw.Draw(img)
        offset = 10
        blob_size = MAX_RADIUS // 4
        draw.ellipse(
            [CENTER[0]-blob_size+offset, CENTER[1]-blob_size+offset,
             CENTER[0]+blob_size+offset, CENTER[1]+blob_size+offset],
            fill=(60, 60, 60)
        )

    def generate_spoiled(img):
        draw = ImageDraw.Draw(img)
        for _ in range(3):
            x_offset = random.randint(-20, 20)
            y_offset = random.randint(-20, 20)
            blob_size = random.randint(20, 40)
            draw.ellipse(
                [CENTER[0]-blob_size+x_offset, CENTER[1]-blob_size+y_offset,
                 CENTER[0]+blob_size+x_offset, CENTER[1]+blob_size+y_offset],
                fill=(50, 50, 50)
            )

    generators = {
        "Fresh": (generate_fresh, "clean_ring"),
        "Water": (generate_water, "weak_ring"),
        "Urea": (generate_urea, "fine_crystals"),
        "Detergent": (generate_detergent, "needle_crystals"),
        "Formalin": (generate_formalin, "dense_dark_deposit"),
        "H2O2": (generate_h2o2, "center_blob"),
        "Spoiled": (generate_spoiled, "dark_deposit")
    }

    for adulterant in ADULTERANTS:
        adulterant_dir = Path(output_dir) / adulterant
        adulterant_dir.mkdir(parents=True, exist_ok=True)
        generator, deposit_type = generators[adulterant]

        for i in range(n_samples_per_class):
            img = Image.new('RGB', (SIZE, SIZE), color=(255, 255, 255))
            generator(img)
            img_path = adulterant_dir / f"{adulterant.lower()}_{i:04d}.png"
            img.save(img_path)
            results.append((str(img_path), deposit_type, adulterant))

    return results

def ensure_directories():
    Path("data/synthetic/vision").mkdir(parents=True, exist_ok=True)
    Path("certificates").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("notebooks").mkdir(parents=True, exist_ok=True)


def generate_dataset_yaml(val_folder: str | Path, save_yaml_path: str | Path) -> Path:
    print("[Stage 1] Writing dataset.yaml ...")
    val_path = Path(val_folder)
    final_yaml_path = Path(save_yaml_path)
    final_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    EXCLUDED_DIR_NAMES = {"images", "labels", "train", "val"}
    detected_classes = sorted(
        f.name for f in val_path.iterdir()
        if f.is_dir() and f.name not in EXCLUDED_DIR_NAMES
    )
    if not detected_classes:
        print(f"  WARNING: No valid class sub-folders found in {val_path}.")

    data_root = val_path.parent
    train_target = "train" if (data_root / "train").exists() else str(data_root / "train")
    val_target = val_path.name if data_root == val_path.parent else str(val_path)

    metadata = {
        "path" : str(data_root),
        "train": train_target,
        "val"  : val_target,
        "names": {idx: name for idx, name in enumerate(detected_classes)},
    }

    with open(final_yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(metadata, fh, default_flow_style=False, sort_keys=False)

    print(f"  Saved  : {final_yaml_path}")
    print(f"  Classes: {detected_classes}")
    return final_yaml_path

# ============================================================================
# DATASET PREPARATION COMPONENT
# ============================================================================
def _auto_prepare_dataset_split(dataset_root: Path, split_ratio: float = 0.8) -> None:
    excluded_names = {"train", "val", "test", "images", "labels", ".ipynb_checkpoints"}
    for checkpoint in dataset_root.rglob(".ipynb_checkpoints"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)

    direct_subdirs = [f for f in dataset_root.iterdir() if f.is_dir() and f.name not in excluded_names]

    if not (dataset_root / "train").exists() and direct_subdirs:
        print(f"Flat directory layout detected in '{dataset_root.name}'. Creating data splits...")
        train_root = dataset_root / "train"
        val_root = dataset_root / "val"
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for class_dir in direct_subdirs:
            class_name = class_dir.name
            images = [
                img for img in class_dir.iterdir()
                if img.is_file() and img.suffix.lower() in valid_extensions
            ]

            if not images:
                continue

            (train_root / class_name).mkdir(parents=True, exist_ok=True)
            (val_root / class_name).mkdir(parents=True, exist_ok=True)

            random.seed(42)
            random.shuffle(images)
            split_idx = int(len(images) * split_ratio)

            for img in images[:split_idx]:
                shutil.move(str(img), str(train_root / class_name / img.name))
            for img in images[split_idx:]:
                shutil.move(str(img), str(val_root / class_name / img.name))

            try:
                class_dir.rmdir()
            except OSError:
                pass
        print("Data split layout generated successfully.")


# ============================================================================
# PIPELINE 1: IMAGE CLASSIFICATION ENGINE
# ============================================================================
def train_yolo_classifier(
    dataset_root: str | Path = VAL_FOLDER,
    output_weights_path: str | Path = YOLO_CLS_WEIGHTS,
    epochs: int = 5,
    img_size: int = 224,
    batch_size: int = 16
) -> None:
    base_dir = Path(dataset_root)
    if not base_dir.exists():
        raise FileNotFoundError(f"Classification dataset folder missing at: {base_dir}")

    _auto_prepare_dataset_split(base_dir)

    print(f"[YOLO Cls] Initiating classification framework on: {base_dir.resolve()}")
    model = YOLO("yolov8s-cls.pt")

    model.train(
        data=str(base_dir),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=YOLO_CLS_PROJECT_NAME,
        name=YOLO_CLS_RUN_NAME,
        exist_ok=True,
        verbose=False,
        device="0" if torch.cuda.is_available() else "cpu",
    )

    if hasattr(model, 'trainer') and model.trainer is not None:
        actual_run_dir = Path(model.trainer.save_dir)
        dynamic_best_weights = actual_run_dir / "weights" / "best.pt"

        if dynamic_best_weights.exists():
            dest = Path(output_weights_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(dynamic_best_weights, dest)
            print(f"[YOLO Cls] Registration complete. Weights deployed to: {dest.resolve()}")
            return

    print("Warning: Dynamic target validation weights could not be collected.")


# ============================================================================
# PIPELINE 2: STANDARD OBJECT DETECTION ENGINE
# ============================================================================
def train_yolo_detector(
    dataset_root: str | Path = VAL_FOLDER,
    yaml_save_path: str | Path = YAML_CONFIG_PATH,
    output_weights_path: str | Path = YOLO_DET_WEIGHTS,
    epochs: int = 5,
    img_size: int = 640,
    batch_size: int = 16
) -> None:
    base_dir = Path(dataset_root)
    final_yaml = Path(yaml_save_path)

    if not base_dir.exists():
        raise FileNotFoundError(f"Detection dataset folder missing at: {base_dir}")

    _auto_prepare_dataset_split(base_dir)

    print(f"[YOLO Det] Mapping configurations out to file: {final_yaml.name}")
    final_yaml.parent.mkdir(parents=True, exist_ok=True)

    train_dir = base_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Structured training directory missing at: {train_dir}")

    detected_classes = sorted([f.name for f in train_dir.iterdir() if f.is_dir()])

    metadata = {
        "path": str(base_dir.resolve()),
        "train": "train",
        "val": "val",
        "names": {idx: name for idx, name in enumerate(detected_classes)},
    }

    with open(final_yaml, "w", encoding="utf-8") as fh:
        yaml.dump(metadata, fh, default_flow_style=False, sort_keys=False)

    print("[YOLO Det] Starting structural object detection execution...")
    model = YOLO("yolov8s.pt")

    model.train(
        data=str(final_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=YOLO_DET_PROJECT_NAME,
        name=YOLO_DET_RUN_NAME,
        exist_ok=True,
        verbose=False,
        device="0" if torch.cuda.is_available() else "cpu",
    )

    if hasattr(model, 'trainer') and model.trainer is not None:
        actual_run_dir = Path(model.trainer.save_dir)
        dynamic_best_weights = actual_run_dir / "weights" / "best.pt"

        if dynamic_best_weights.exists():
            dest = Path(output_weights_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(dynamic_best_weights, dest)
            print(f"[YOLO Det] Registration complete. Weights deployed to: {dest.resolve()}")
            return

    print("Warning: Dynamic target validation weights could not be collected.")


# ============================================================================
# UNIFIED PREDICTOR CLASS
# ============================================================================
class MilkVisionPredictor:
    """
    Unified inference pipeline combining YOLO Classification and Object Detection.
    Provides execution flags to dynamically control database entries and JSON logging.
    """
    def __init__(
        self,
        cls_model_path: Optional[str] = None,
        det_model_path: Optional[str] = None,
        db_path: str = DB_PATH
    ) -> None:
        """
        Initialize the predictor tracking engines and targeted database parameters.
        Defaults to centralized system constants if arguments are omitted.
        """
        resolved_cls_path = cls_model_path if cls_model_path else YOLO_CLS_WEIGHTS
        resolved_det_path = det_model_path if det_model_path else YOLO_DET_WEIGHTS

        self.cls_model = YOLO(resolved_cls_path) if resolved_cls_path and Path(resolved_cls_path).exists() else None
        self.det_model = YOLO(resolved_det_path) if resolved_det_path and Path(resolved_det_path).exists() else None
        self.db_path = db_path

    def predict(
        self,
        image_path: str,
        output_dir: str = JSON_OUTPUT_FOLDER,
        save_db: bool = False,
        save_json: bool = False
    ) -> Dict[str, Any]:
        """
        Execute combined multi-model inference on a target image file path.
        """
        test_img_path = Path(image_path)
        if not test_img_path.exists():
            raise FileNotFoundError(f"[Error] Target image file not found at: {test_img_path}")

        try:
            data_sample_id, batch_id = test_img_path.stem.split('_')
        except ValueError:
            data_sample_id = test_img_path.stem
            batch_id = "UNKNOWN_BATCH"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        combined_payload = {
            "data_sample_id": data_sample_id,
            "batch_id": batch_id,
            "image_filename": str(test_img_path.resolve()),
            "created_at": timestamp,
            "classification_analysis": None,
            "object_detection_analysis": None
        }

        # --------------------------------------------------------------------
        # Modality 1: Image Classification
        # --------------------------------------------------------------------
        if self.cls_model:
            cls_result = self.cls_model(str(test_img_path), verbose=False)[0]

            if cls_result.probs is not None:
                raw_probs = cls_result.probs.data.cpu().numpy()
                class_names = cls_result.names

                all_class_scores = {}
                detected_classes = []

                for idx, confidence in enumerate(raw_probs):
                    c_name = class_names[idx]
                    percentage = round(float(confidence) * 100, 2)
                    all_class_scores[c_name] = f"{percentage}%"
                    if percentage > 5.0:
                        detected_classes.append(c_name)

                top_conf = round(float(cls_result.probs.top1conf.item()), 4)
                top_class = cls_result.names[cls_result.probs.top1]

                cls_payload = {
                    "predicted_class": top_class,
                    "confidence": top_conf,
                    "class_scores": all_class_scores,
                    "detected_mixture_components": detected_classes
                }
                combined_payload["classification_analysis"] = cls_payload
            else:
                print("[Warning] Classification model returned no probabilities. Check if a detection model was loaded by mistake.")

        # --------------------------------------------------------------------
        # Modality 2: Object Detection
        # --------------------------------------------------------------------
        if self.det_model:
            det_result = self.det_model(str(test_img_path), verbose=False)[0]

            if det_result.boxes is not None:
                boxes = det_result.boxes

                detected_objects = []
                object_counts = {}

                for box in boxes:
                    cls_idx = int(box.cls.item())
                    obj_name = det_result.names[cls_idx]
                    box_conf = round(float(box.conf.item()), 4)

                    # FIX: Kept explicit [0] index map array slicing resolution intact
                    xyxy_coords = box.xyxy[0].cpu().numpy().tolist()

                    detected_objects.append({
                        "object_class": obj_name,
                        "confidence": box_conf,
                        "bounding_box_xyxy": [round(coord, 2) for coord in xyxy_coords]
                    })

                    object_counts[obj_name] = object_counts.get(obj_name, 0) + 1

                det_payload = {
                    "total_objects_detected": len(detected_objects),
                    "object_class_counts": object_counts,
                    "detailed_instances": detected_objects
                }
                combined_payload["object_detection_analysis"] = det_payload
            else:
                print("[Warning] Detection model returned no boxes. Check if a classification model was loaded by mistake.")

        # --------------------------------------------------------------------
        # CHANGE: Unified Database Persistence Logic
        # Gathers parameters from both prediction branches and transforms elements
        # into serialized strings to safely fit the new text column mappings.
        # --------------------------------------------------------------------
        if save_db:
            try:
                cls_info = combined_payload.get("classification_analysis") or {}
                det_info = combined_payload.get("object_detection_analysis") or {}

                deposit_type_val = json.dumps(cls_info.get("detected_mixture_components", [])) if cls_info else "[]"
                pred_class_val = cls_info.get("predicted_class", "Unknown") if cls_info else "Unknown"
                conf_val = str(cls_info.get("confidence", "0.0")) if cls_info else "0.0"
                det_info_str = json.dumps(det_info) if det_info else "{}"

                insert_vision_analysis(
                    sample_id=data_sample_id,
                    batch_id=batch_id,
                    image_path=str(test_img_path.resolve()),
                    deposit_type=deposit_type_val,
                    predicted_class=pred_class_val,
                    confidence=conf_val,
                    object_detection_info=det_info_str,
                    db_path=self.db_path
                )
            except Exception as e:
                print(f"[Warning] Database insertion fault on combined mapping step: {e}")

        # --------------------------------------------------------------------
        # Data Persistence: Serialization Out
        # --------------------------------------------------------------------
        if save_json:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            json_target_path = out_path / f"{data_sample_id}_{batch_id}_combined.json"
            with open(json_target_path, "w", encoding="utf-8") as json_file:
                json.dump(combined_payload, json_file, indent=4)

        return combined_payload

if __name__ == "__main__":
    # Test: initialize DB
    init_db()
    ensure_directories()
    print("[OK] Library initialized successfully")