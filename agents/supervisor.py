# Supervisor Agent
# Role: Entry point, ID generation, state initialization

from agents.state import FoodInvestigationState
from utils.ids import batch_id, vision_id, aroma_id, taste_id, corr_id, inv_id, pass_id, cert_id
from models.database import get_session
from models.db import Batch


def supervisor_node(state: FoodInvestigationState) -> FoodInvestigationState:
    """Initialize batch record and generate all IDs."""
    batch_id_val = batch_id()

    # Persist batch to SQLite
    session = get_session()
    try:
        batch_record = Batch(
            id=batch_id_val,
            food_type=state["food_type"],
            batch_name=state["batch_name"]
        )
        session.add(batch_record)
        session.commit()
    finally:
        session.close()

    return {
        **state,
        "batch_id": batch_id_val,
        "vision_id": vision_id(),
        "aroma_id": aroma_id(),
        "taste_id": taste_id(),
        "correlation_id": corr_id(),
        "investigation_id": inv_id(),
        "passport_id": pass_id(),
        "certificate_id": cert_id(),
        "errors": [],
    }
