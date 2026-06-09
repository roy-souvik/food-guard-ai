from langgraph.graph import StateGraph, END
from agents.state import FoodInvestigationState
from utils.ids import batch_id, vision_id, aroma_id, taste_id, corr_id, inv_id, pass_id, cert_id


# --- Agent stubs (will be replaced phase by phase) ---

def supervisor_node(state: FoodInvestigationState) -> FoodInvestigationState:
    return {
        **state,
        "batch_id": batch_id(),
        "vision_id": vision_id(),
        "aroma_id": aroma_id(),
        "taste_id": taste_id(),
        "correlation_id": corr_id(),
        "investigation_id": inv_id(),
        "passport_id": pass_id(),
        "certificate_id": cert_id(),
        "errors": [],
    }


def vision_node(state: FoodInvestigationState) -> FoodInvestigationState:
    return {**state, "vision_result": {"score": 0, "label": "pending", "findings": []}}


def aroma_node(state: FoodInvestigationState) -> FoodInvestigationState:
    return {**state, "aroma_result": {"score": 0, "label": "pending", "findings": []}}


def taste_node(state: FoodInvestigationState) -> FoodInvestigationState:
    return {**state, "taste_result": {"score": 0, "label": "pending", "findings": []}}


def correlation_node(state: FoodInvestigationState) -> FoodInvestigationState:
    return {**state, "correlation_result": {"pattern_type": "pending", "confidence_delta": 0}}


def food_safety_node(state: FoodInvestigationState) -> FoodInvestigationState:
    return {**state, "investigation_result": {"verdict": "pending", "score": 0}}


def reviewer_node(state: FoodInvestigationState) -> FoodInvestigationState:
    return {**state, "review_result": {"confidence_adjustment": 0, "notes": []}}


def passport_node(state: FoodInvestigationState) -> FoodInvestigationState:
    return {**state, "passport_result": {"status": "pending"}}


# --- Graph assembly ---

def build_graph():
    g = StateGraph(FoodInvestigationState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("vision", vision_node)
    g.add_node("aroma", aroma_node)
    g.add_node("taste", taste_node)
    g.add_node("correlation", correlation_node)
    g.add_node("food_safety", food_safety_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("passport", passport_node)

    g.set_entry_point("supervisor")

    # Supervisor → parallel analysis
    g.add_edge("supervisor", "vision")
    g.add_edge("supervisor", "aroma")
    g.add_edge("supervisor", "taste")

    # All three → correlation
    g.add_edge("vision", "correlation")
    g.add_edge("aroma", "correlation")
    g.add_edge("taste", "correlation")

    # Linear from correlation onward
    g.add_edge("correlation", "food_safety")
    g.add_edge("food_safety", "reviewer")
    g.add_edge("reviewer", "passport")
    g.add_edge("passport", END)

    return g.compile()


graph = build_graph()
