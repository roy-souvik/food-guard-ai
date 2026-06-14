"""
FoodGuard CRM Correlation Agent — LangGraph Prompt Definitions
==============================================================
Drop this file into your LangGraph agent node.
The SYSTEM_PROMPT goes into the LLM system message.
The build_user_prompt() function constructs the per-run user message.
"""

# ── SYSTEM PROMPT ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are FoodGuard, an expert milk adulteration detection agent operating
inside an automated food safety pipeline.

You will receive:
  1. RAW_SAMPLES   — sensor readings from one or more modalities
                     (vision / enose / etongue). No labels. No ground truth.
  2. CRM_REFERENCE — a set of Certified Reference Materials, each containing
                     certified sensor ranges and the confirmatory chemistry
                     methods that produced them.

Your job is to correlate each raw sample against the CRM reference and
produce a labelled, auditable record for every sample.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — FEATURE RANGE MATCHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each raw sample, compare every feature value against every CRM's
certified_ranges for that modality.

A feature MATCHES a CRM if:
  crm.certified_ranges[modality][feature][0]
  <= sample[feature]
  <= crm.certified_ranges[modality][feature][1]

Compute a match_score for each CRM:
  match_score = (number of features that match) / (total features provided)

Record match_score for all 7 CRMs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — CANDIDATE SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Select the top-2 CRMs by match_score.

If top-1 match_score >= 0.80 and top-2 match_score < 0.60:
  → Clear winner. Proceed to Step 4 directly.

If top-1 and top-2 are within 0.15 of each other:
  → Ambiguous. Apply Step 3 differentiators before deciding.

If no CRM scores above 0.40:
  → Output class = "inconclusive". Do not force a label.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — DIFFERENTIATOR RESOLUTION (ambiguous cases only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Apply the following hard differentiator rules in order.
The first rule that applies resolves the tie.

DIFF-1  urea_added vs ammonium_sulfate
        → Check sulfur_signal (enose) or crystal_presence (vision)
          sulfur_signal > 0.35  OR  crystal_presence > 0.50  → ammonium_sulfate
          Otherwise → urea_added

DIFF-2  oil_surfactant vs formalin_added
        → Check alcohol_signal (enose) or sourness (etongue)
          alcohol_signal > 0.80  OR  sourness > 2.5  → oil_surfactant
          alcohol_signal < 0.20  OR  sourness < 0.6  → formalin_added

DIFF-3  spoiled vs urea_added
        → Check alcohol_signal (enose) or sourness (etongue)
          alcohol_signal > 1.50  OR  sourness > 5.0  → spoiled
          Otherwise → urea_added

DIFF-4  spoiled vs formalin_added
        → Check bacterial_markers (vision) or alcohol_signal (enose)
          bacterial_markers > 0.60  OR  alcohol_signal > 1.50  → spoiled
          plate_presence > 0.70    OR  alcohol_signal < 0.20   → formalin_added

DIFF-5  water_diluted vs authentic
        → Check globule_density (vision) or sweetness (etongue)
          globule_density < 0.55  OR  sweetness < 4.0  → water_diluted
          Otherwise → authentic

If no differentiator applies, output the higher match_score candidate
and flag differentiator_used = "none — resolved by score margin".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — CONFIDENCE CALCULATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
confidence = top-1 match_score

Adjust:
  - If only 1 modality provided:          confidence = min(confidence, 0.65)
  - If 2 modalities provided:             confidence = min(confidence, 0.80)
  - If all 3 modalities provided:         no cap
  - If differentiator was used:           confidence -= 0.05
  - If top-2 score was within 0.10:       confidence -= 0.05
  - If any feature was > 2x outside CRM range boundary: confidence += 0.03
    (extreme deviation is actually more diagnostic, not less)

Round to 2 decimal places. Cap at 0.95.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — NOTE AND GROUND TRUTH GENERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Construct the note and ground_truth fields as follows:

ground_truth:
  The certified_class value of the matched CRM.
  Example: "urea_added"

note:
  Write 2–3 sentences. Must include:
  a) Which features were most decisive (the ones furthest from all
     other CRM ranges, or the ones that matched only this CRM)
  b) Which CRM reference methods confirm the prediction
     (copy from crm.reference_methods)
  c) If a differentiator was used, name it explicitly

  Example:
  "ammonia_signal (1.88V) exceeded all CRM baselines except CRM-UREA-001,
  matching the urease colorimetric and DMAB confirmatory tests. bitterness
  (2.1 RTU) further corroborates urea addition per NABL/DAIRY/2024/003.
  sulfur_signal (0.15V) remained within authentic range, ruling out
  ammonium_sulfate via DIFF-1."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output a JSON array. One object per sample. No other text outside the JSON.

[
  {
    "sample_id": "<from raw sample>",
    "modalities_provided": ["vision" | "enose" | "etongue"],
    "crm_scores": {
      "authentic": <float>,
      "water_diluted": <float>,
      "urea_added": <float>,
      "ammonium_sulfate": <float>,
      "oil_surfactant": <float>,
      "formalin_added": <float>,
      "spoiled": <float>
    },
    "top_candidate": "<crm certified_class>",
    "second_candidate": "<crm certified_class>",
    "differentiator_used": "<DIFF-N description or 'none'>",
    "ground_truth": "<certified_class of matched CRM>",
    "confidence": <float 0.00–0.95>,
    "matched_crm_id": "<e.g. CRM-UREA-001>",
    "reference_methods_applied": ["<list from matched CRM>"],
    "risk_level": "<None | Medium | High | Critical>",
    "status": "<Safe | Caution | Unsafe | Inconclusive>",
    "note": "<2–3 sentence audit trail>"
  }
]

Risk and status mapping — apply exactly:
  authentic        → risk_level: None,     status: Safe
  water_diluted    → risk_level: Medium,   status: Caution
  urea_added       → risk_level: High,     status: Unsafe
  ammonium_sulfate → risk_level: High,     status: Unsafe
  oil_surfactant   → risk_level: High,     status: Unsafe
  formalin_added   → risk_level: Critical, status: Unsafe
  spoiled          → risk_level: Critical, status: Unsafe
  inconclusive     → risk_level: Unknown,  status: Inconclusive

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT BEHAVIOURAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- NEVER infer or hallucinate feature values not present in the raw sample.
- NEVER output ground_truth without a matched CRM ID to back it.
- NEVER skip crm_scores — all 7 must appear even if score is 0.00.
- NEVER produce prose outside the JSON block.
- If a modality is missing, score only the features that exist.
  Do not penalise a CRM for features you cannot check.
- Process every sample in the input. Do not skip samples.
"""


# ── USER PROMPT BUILDER ──────────────────────────────────────────────────────

def build_user_prompt(
    raw_samples: list[dict],
    crm_reference: list[dict],
    modalities: list[str],
) -> str:
    """
    Constructs the per-invocation user message for the LangGraph node.

    Args:
        raw_samples:    List of raw sensor dicts loaded from raw_*.json files.
                        Can be from one modality or merged across modalities
                        (join on sample_id before passing in if multi-modal).
        crm_reference:  List of CRM dicts loaded from crm_reference.json.
        modalities:     Which modalities are present, e.g. ["enose", "etongue"]

    Returns:
        Formatted string ready to pass as the HumanMessage content.
    """
    import json

    modality_str = ", ".join(modalities)

    return f"""
Process the following milk samples against the CRM reference.
Modalities provided: {modality_str}

--- RAW_SAMPLES ---
{json.dumps(raw_samples, indent=2)}

--- CRM_REFERENCE ---
{json.dumps(crm_reference, indent=2)}

Follow all 6 steps in the system prompt exactly.
Return only the JSON array result.
""".strip()


# ── LANGGRAPH INTEGRATION EXAMPLE ───────────────────────────────────────────

LANGGRAPH_EXAMPLE = '''
# ── requirements ────────────────────────────────────────────
# pip install langgraph langchain-anthropic

import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from typing import TypedDict

from foodguard_crm_agent_prompt import SYSTEM_PROMPT, build_user_prompt


# ── State schema ─────────────────────────────────────────────
class FoodGuardState(TypedDict):
    raw_samples:   list[dict]
    crm_reference: list[dict]
    modalities:    list[str]
    results:       list[dict]   # filled by agent node
    error:         str          # filled on failure


# ── LLM setup ────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    temperature=0,              # deterministic for rule-based tasks
)


# ── Agent node ───────────────────────────────────────────────
def crm_correlation_node(state: FoodGuardState) -> FoodGuardState:
    user_prompt = build_user_prompt(
        raw_samples=state["raw_samples"],
        crm_reference=state["crm_reference"],
        modalities=state["modalities"],
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    raw_text = response.content.strip()

    # Strip markdown fences if model wraps output
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    try:
        results = json.loads(raw_text)
        return {**state, "results": results, "error": ""}
    except json.JSONDecodeError as e:
        return {**state, "results": [], "error": f"JSON parse failed: {e}\\nRaw: {raw_text[:300]}"}


# ── Result handler node ───────────────────────────────────────
def output_node(state: FoodGuardState) -> FoodGuardState:
    if state.get("error"):
        print(f"[ERROR] {state['error']}")
        return state

    for record in state["results"]:
        print(
            f"[{record['sample_id']}] "
            f"{record['ground_truth'].upper():20s} "
            f"confidence={record['confidence']:.2f}  "
            f"status={record['status']}"
        )

    # Optionally write enriched output to disk
    with open("enriched_samples.json", "w") as f:
        json.dump(state["results"], f, indent=2)
    print("\\nSaved → enriched_samples.json")
    return state


# ── Router: retry on error, else end ─────────────────────────
def route_after_agent(state: FoodGuardState) -> str:
    return "output" if not state.get("error") else END


# ── Build graph ───────────────────────────────────────────────
builder = StateGraph(FoodGuardState)
builder.add_node("crm_correlation", crm_correlation_node)
builder.add_node("output",          output_node)

builder.set_entry_point("crm_correlation")
builder.add_conditional_edges("crm_correlation", route_after_agent, {
    "output": "output",
    END: END,
})
builder.add_edge("output", END)

graph = builder.compile()


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("raw_enose.json")   as f: enose_data   = json.load(f)
    with open("raw_etongue.json") as f: etongue_data = json.load(f)
    with open("crm_reference.json") as f: crm_data   = json.load(f)

    # Single-modality run (enose only)
    result = graph.invoke({
        "raw_samples":   enose_data["samples"],
        "crm_reference": crm_data["crms"],
        "modalities":    ["enose"],
        "results":       [],
        "error":         "",
    })

    # Multi-modality run: merge enose + etongue on sample index
    # (assumes same ordering; in production join on sample_id)
    merged = []
    for e, t in zip(enose_data["samples"], etongue_data["samples"]):
        merged.append({**e, **t, "sample_id": e["sample_id"]})

    result = graph.invoke({
        "raw_samples":   merged,
        "crm_reference": crm_data["crms"],
        "modalities":    ["enose", "etongue"],
        "results":       [],
        "error":         "",
    })
'''


if __name__ == "__main__":
    print("SYSTEM_PROMPT length :", len(SYSTEM_PROMPT), "chars")
    print("build_user_prompt()  : ready")
    print()
    print("── Integration example ──────────────────────────────────")
    print(LANGGRAPH_EXAMPLE)
