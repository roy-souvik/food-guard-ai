from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, JSON, Column
import sqlalchemy as sa


class Batch(SQLModel, table=True):
    id: str = Field(primary_key=True)
    food_type: str
    batch_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VisionAnalysis(SQLModel, table=True):
    __tablename__ = "vision_analysis"
    id: str = Field(primary_key=True)
    batch_id: str
    image_path: Optional[str] = None
    model_name: str = "YOLOv12"
    score: float
    label: str
    findings: Optional[str] = None  # JSON string
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AromaAnalysis(SQLModel, table=True):
    __tablename__ = "aroma_analysis"
    id: str = Field(primary_key=True)
    batch_id: str
    sensor_data: Optional[str] = None  # JSON string
    score: float
    label: str
    findings: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TasteAnalysis(SQLModel, table=True):
    __tablename__ = "taste_analysis"
    id: str = Field(primary_key=True)
    batch_id: str
    sensor_data: Optional[str] = None
    score: float
    label: str
    findings: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CorrelationRule(SQLModel, table=True):
    __tablename__ = "correlation_rules"
    rule_id: str = Field(primary_key=True)
    adulterant: str
    vision_signal: str        # comma-separated tokens or 'any'
    aroma_signal: str
    taste_signal: str
    pattern_type: str         # convergent|contradiction|partial|masked|multi_adulterant|unknown_pattern|clean
    confidence_delta: float
    severity: str             # high|medium|low
    explanation: str
    recommended_action: str


class Correlation(SQLModel, table=True):
    __tablename__ = "correlations"
    id: str = Field(primary_key=True)
    batch_id: str
    vision_id: str
    aroma_id: str
    taste_id: str
    matched_rules: Optional[str] = None   # JSON string
    pattern_type: str
    suspected_adulterant: str
    confidence_delta: float
    contradictions: Optional[str] = None  # JSON string
    recommended_action: str
    correlation_summary: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Investigation(SQLModel, table=True):
    __tablename__ = "investigations"
    id: str = Field(primary_key=True)
    batch_id: str
    vision_id: str
    aroma_id: str
    taste_id: str
    correlation_id: str
    overall_score: float
    verdict: str
    suspected_adulterant: str
    risk_level: str
    reasoning: Optional[str] = None  # JSON string
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Passport(SQLModel, table=True):
    __tablename__ = "passports"
    id: str = Field(primary_key=True)
    investigation_id: str
    certificate_id: str
    certificate_hash: str
    passport_json: Optional[str] = None  # full JSON
    qr_code_path: Optional[str] = None
    blockchain_status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
