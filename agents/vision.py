# Vision Agent
# Role: YOLOv12 image analysis for milk adulteration detection

from agents.state import FoodInvestigationState
from models.database import get_session
from models.db import VisionAnalysis


def vision_node(state: FoodInvestigationState) -> FoodInvestigationState:
    """Analyze milk image using YOLOv12 (Phase 3 training output)."""
    # TODO: Load trained YOLOv12 model
    # TODO: Run inference on state["image_path"]
    # TODO: Extract class, confidence, findings

    # Stub for now
    vision_result = {
        "class": "authentic",
        "score": 0.0,
        "findings": []
    }

    # Persist to SQLite
    session = get_session()
    try:
        analysis = VisionAnalysis(
            id=state["vision_id"],
            batch_id=state["batch_id"],
            image_path=state.get("image_path"),
            score=vision_result["score"],
            label=vision_result["class"],
            findings=str(vision_result["findings"])
        )
        session.add(analysis)
        session.commit()
    finally:
        session.close()

    return {**state, "vision_result": vision_result}
