# Aroma Agent
# Role: E-Nose sensor analysis via XGBoost for volatile compound detection

from agents.state import FoodInvestigationState
from models.database import get_session
from models.db import AromaAnalysis


def aroma_node(state: FoodInvestigationState) -> FoodInvestigationState:
    """Analyze aroma profile using XGBoost on E-Nose sensor data (Phase 3 training output)."""
    # TODO: Load trained XGBoost model (enose)
    # TODO: Use state["aroma_data"] = {sensor_1: val, sensor_2: val, ...}
    # TODO: Run prediction on sensor array

    # Stub for now
    aroma_result = {
        "class": "authentic",
        "score": 0.0,
        "findings": []
    }

    # Persist to SQLite
    session = get_session()
    try:
        analysis = AromaAnalysis(
            id=state["aroma_id"],
            batch_id=state["batch_id"],
            sensor_data=str(state.get("aroma_data", {})),
            score=aroma_result["score"],
            label=aroma_result["class"],
            findings=str(aroma_result["findings"])
        )
        session.add(analysis)
        session.commit()
    finally:
        session.close()

    return {**state, "aroma_result": aroma_result}
