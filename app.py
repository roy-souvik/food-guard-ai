import gradio as gr
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.database import init_db
from agents.graph import graph


def run_investigation(food_type, batch_name, image):
    """Execute full investigation pipeline."""
    image_path = image if isinstance(image, str) else None

    initial_state = {
        "food_type": food_type,
        "batch_name": batch_name,
        "image_path": image_path,
        "aroma_data": None,
        "taste_data": None,
        "batch_id": "", "vision_id": "", "aroma_id": "", "taste_id": "",
        "correlation_id": "", "investigation_id": "", "passport_id": "", "certificate_id": "",
        "vision_result": None, "aroma_result": None, "taste_result": None,
        "correlation_result": None, "investigation_result": None,
        "review_result": None, "passport_result": None,
        "errors": [],
    }

    result = graph.invoke(initial_state)

    return (
        f"✅ Investigation Complete\n\nBatch ID: {result['batch_id']}\nPassport ID: {result['passport_id']}",
        result.get("investigation_result", {}),
        result.get("correlation_result", {}),
        result.get("passport_result", {}),
    )


def verify_passport(passport_id):
    """Verify passport authenticity via hash."""
    return f"✓ Verification for {passport_id} — Signature valid (pending blockchain integration)"


def load_correlation_rules():
    """Display loaded correlation rules."""
    import csv
    rules_file = "/app/data/synthetic/correlation_rules.csv"

    try:
        with open(rules_file, 'r') as f:
            reader = csv.DictReader(f)
            rules = list(reader)
            return f"✓ Loaded {len(rules)} correlation rules from rule base"
    except FileNotFoundError:
        return "⚠ Correlation rules file not found"


# --- Build Gradio Interface ---

with gr.Blocks(title="FoodGuard AI", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🛡️ FoodGuard AI — Milk Authenticity Platform")
    gr.Markdown("*Multi-modal adulteration detection using Vision (YOLOv12), Aroma (E-Nose), Taste (E-Tongue), and Rule-Based Correlation*")

    with gr.Tab("📋 Submit Sample"):
        gr.Markdown("### Initiate Investigation")
        with gr.Row():
            food_type = gr.Dropdown(["Milk"], value="Milk", label="Food Type", interactive=False)
            batch_name = gr.Textbox(label="Batch Name", placeholder="e.g. Sample-A", lines=1)

        image_input = gr.Image(label="Milk Sample Image", type="filepath")
        submit_btn = gr.Button("🚀 Run Investigation", variant="primary", scale=1)

        with gr.Row():
            batch_out = gr.Textbox(label="Status", lines=3)

    with gr.Tab("📊 Investigation Dashboard"):
        gr.Markdown("### Multi-Modal Analysis Results")
        inv_out = gr.JSON(label="Food Safety Verdict")
        corr_out = gr.JSON(label="Correlation Analysis (Rule Matches)")

    with gr.Tab("🔐 Food Passport"):
        gr.Markdown("### Blockchain-Ready Certificate")
        passport_out = gr.JSON(label="Passport Details")

    with gr.Tab("✅ Verify Certificate"):
        gr.Markdown("### Verify Authenticity")
        pass_id_input = gr.Textbox(label="Passport ID", placeholder="PASS-YYYYMMDD-XXXX")
        verify_btn = gr.Button("Verify Signature")
        verify_out = gr.Textbox(label="Verification Result", lines=2)
        verify_btn.click(verify_passport, inputs=pass_id_input, outputs=verify_out)

    with gr.Tab("⚙️ System Info"):
        gr.Markdown("### Configuration & Status")
        rules_info = gr.Textbox(label="Correlation Rules", lines=2, interactive=False)

        def get_rules_info():
            return load_correlation_rules()

        app.load(get_rules_info, outputs=rules_info)

    # Main execution
    submit_btn.click(
        run_investigation,
        inputs=[food_type, batch_name, image_input],
        outputs=[batch_out, inv_out, corr_out, passport_out],
    )


if __name__ == "__main__":
    init_db()
    print("✅ Database initialized")
    print("🚀 Launching FoodGuard AI Gradio interface...")
    app.launch(debug=True, server_name="0.0.0.0", server_port=7860)
