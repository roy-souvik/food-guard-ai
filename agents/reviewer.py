# Phase 5: Reviewer Agent
# Role: Confidence adjustment and edge-case escalation

from agents.state import FoodInvestigationState


def reviewer_node(state: FoodInvestigationState) -> FoodInvestigationState:
    """Challenge and adjust confidence based on weak signals or contradictions."""
    # TODO: Analyze all prior results
    # TODO: Flag edge cases (low consensus, borderline scores)
    # TODO: Adjust confidence_adjustment based on rule strength
    # TODO: Escalation logic for ambiguous cases

    # Stub for now
    review_result = {
        "final_confidence": state["investigation_result"].get("authenticity_score", 0),
        "confidence_adjustment": 0,
        "notes": [],
        "escalation_needed": False
    }

    return {**state, "review_result": review_result}
