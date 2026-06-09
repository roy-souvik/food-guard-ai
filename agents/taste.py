# Taste Agent
# Role: E-Tongue sensor analysis via XGBoost for taste profile detection

from agents.state import FoodInvestigationState
from models.database import get_session
from models.db import TasteAnalysis


def taste_node(state: FoodInvestigationState) -> FoodInvestigationState:
    """Analyze taste profile using XGBoost on E-Tongue sensor data (Phase 3 training output)."""
    # TODO: Load trained XGBoost model (etongue)
    # TODO: Use state["taste_data"] = {sweetness: val, bitterness: val, ...}
    # TODO: Run prediction on taste sensor array

    # Stub for now
    taste_result = {
        "class": "authentic",
        "score": 0.0,
        "findings": []
    }

    # Persist to SQLite
    session = get_session()
    try:
        analysis = TasteAnalysis(
            id=state["taste_id"],
            batch_id=state["batch_id"],
            sensor_data=str(state.get("taste_data", {})),
            score=taste_result["score"],
            label=taste_result["class"],
            findings=str(taste_result["findings"])
        )
        session.add(analysis)
        session.commit()
    finally:
        session.close()

    return {**state, "taste_result": taste_result}
