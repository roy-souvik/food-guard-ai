# Phase 6: Passport Agent
# Role: Certificate generation, hash, QR code, blockchain stub

import json
import hashlib
import qrcode
from io import BytesIO
import base64
from agents.state import FoodInvestigationState
from models.database import get_session
from models.db import Passport


def generate_certificate_hash(passport_data: dict) -> str:
    """Generate SHA-256 hash of canonical passport JSON."""
    canonical = json.dumps(passport_data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def generate_qr_code(passport_id: str, certificate_hash: str) -> str:
    """Generate QR code encoding passport_id + certificate_hash."""
    qr_data = f"{passport_id}:{certificate_hash}"
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image()

    # Convert to base64
    buf = BytesIO()
    img.save(buf, format='PNG')
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return img_base64


def passport_node(state: FoodInvestigationState) -> FoodInvestigationState:
    """Generate full passport with certificate, hash, QR, blockchain stub."""

    # Build full passport payload
    passport_data = {
        "passport_id": state["passport_id"],
        "certificate_id": state["certificate_id"],
        "batch": {
            "id": state["batch_id"],
            "food_type": state["food_type"],
            "batch_name": state["batch_name"]
        },
        "analysis": {
            "vision": state.get("vision_result", {}),
            "aroma": state.get("aroma_result", {}),
            "taste": state.get("taste_result", {}),
            "correlation": state.get("correlation_result", {})
        },
        "investigation": state.get("investigation_result", {}),
        "review": state.get("review_result", {})
    }

    # Generate hash and QR
    cert_hash = generate_certificate_hash(passport_data)
    qr_base64 = generate_qr_code(state["passport_id"], cert_hash)

    # Add certificate metadata
    passport_data["certificate"] = {
        "id": state["certificate_id"],
        "hash_algorithm": "SHA-256",
        "certificate_hash": cert_hash,
        "blockchain_status": "pending",
        "qr_code": qr_base64
    }

    # Persist to SQLite
    session = get_session()
    try:
        passport_record = Passport(
            id=state["passport_id"],
            investigation_id=state["investigation_id"],
            certificate_id=state["certificate_id"],
            certificate_hash=cert_hash,
            passport_json=json.dumps(passport_data),
            qr_code_path="generated",
            blockchain_status="pending"
        )
        session.add(passport_record)
        session.commit()
    finally:
        session.close()

    passport_result = {
        "status": "generated",
        "passport_id": state["passport_id"],
        "certificate_hash": cert_hash,
        "qr_code": qr_base64,
        "blockchain_status": "pending"
    }

    return {**state, "passport_result": passport_result}
