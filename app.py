from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.config import ARTIFACT_DIR, CLASS_NAMES, IMAGE_SIZE
from src.inference import gradcam, load_student, predict_image


# -----------------------------------------------------------------------------
# Page configuration and visual theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WasteWise AI | Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

      :root {
        color-scheme: light;
        --ww-ink: #172a20;
        --ww-muted: #53665b;
        --ww-green: #155b3e;
        --ww-border: #d7e3da;
      }

      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp {
        background: radial-gradient(circle at 8% 0%, #e7f5ef 0, #f5f8f6 34%, #f4f6f9 100%);
        color: var(--ww-ink) !important;
      }
      [data-testid="stAppViewContainer"], [data-testid="stMain"] { background-color: transparent; }
      .block-container { max-width: 1480px; padding-top: 1.6rem; padding-bottom: 3rem; }

      /* Main-page text: dark copy on a pale canvas. */
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stMarkdownContainer"] p,
      .stApp [data-testid="stMarkdownContainer"] li,
      .stApp [data-testid="stMarkdownContainer"] span,
      .stApp label { color: var(--ww-ink); }
      .stApp [data-testid="stCaptionContainer"], .stApp .stCaption {
        color: var(--ww-muted) !important;
      }

      /* Sidebar: dark panel and dark metric cards. */
      [data-testid="stSidebar"] {
        background: #10271f;
        border-right: 1px solid #29483a;
      }
      [data-testid="stSidebar"] * { color: #edf7f1 !important; }
      [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #1c3b2f !important;
        border: 1px solid #426654 !important;
        border-radius: 14px;
        padding: 13px 15px;
        box-shadow: none;
      }
      [data-testid="stSidebar"] [data-testid="stMetricLabel"],
      [data-testid="stSidebar"] [data-testid="stMetricLabel"] *,
      [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
        color: #d3e5da !important;
        opacity: 1 !important;
      }
      [data-testid="stSidebar"] [data-testid="stMetricValue"],
      [data-testid="stSidebar"] [data-testid="stMetricValue"] * {
        color: #ffffff !important;
        opacity: 1 !important;
      }
      [data-testid="stSidebar"] [data-testid="stMetricDelta"] * { color: #d3e5da !important; }

      /* Main metric cards: white surface, dark labels and green values. */
      [data-testid="stAppViewContainer"] [data-testid="stMetric"],
      [data-testid="stMain"] [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #dbe6df !important;
        border-radius: 16px;
        padding: 15px 17px;
        box-shadow: 0 5px 18px rgba(22, 48, 37, .045);
      }
      [data-testid="stAppViewContainer"] [data-testid="stMetricLabel"],
      [data-testid="stMain"] [data-testid="stMetricLabel"],
      [data-testid="stAppViewContainer"] [data-testid="stMetricLabel"] *,
      [data-testid="stMain"] [data-testid="stMetricLabel"] * {
        color: #53665b !important;
        opacity: 1 !important;
      }
      [data-testid="stAppViewContainer"] [data-testid="stMetricValue"],
      [data-testid="stMain"] [data-testid="stMetricValue"],
      [data-testid="stAppViewContainer"] [data-testid="stMetricValue"] *,
      [data-testid="stMain"] [data-testid="stMetricValue"] * {
        color: #155b3e !important;
        opacity: 1 !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
      }

      /* Inputs, uploader, buttons and tabs. */
      div[data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: 1px dashed #82a994 !important;
        border-radius: 16px;
      }
      div[data-testid="stFileUploader"] section *,
      div[data-testid="stCameraInput"] label,
      .stSelectbox label, .stTextInput label, .stFileUploader label {
        color: #20382b !important;
      }
      div.stButton > button[kind="primary"] {
        background: #176b4b;
        border: 1px solid #176b4b;
        border-radius: 11px;
        color: #ffffff !important;
        font-weight: 700;
        min-height: 44px;
      }
      div.stButton > button[kind="primary"] * { color: #ffffff !important; }
      div.stButton > button[kind="primary"]:hover { background: #10583d; border-color: #10583d; }
      div.stButton > button:not([kind="primary"]) {
        color: #173c2c !important;
        background: #ffffff;
        border-color: #b7cbbd;
      }
      div[data-testid="stTabs"] button { color: #294638 !important; font-weight: 700; }
      div[data-testid="stTabs"] button[aria-selected="true"] { color: #125b3c !important; }

      /* Dashboard components. */
      .hero {
        background: linear-gradient(115deg, #123a2b 0%, #176b4b 58%, #3a9870 100%);
        border-radius: 22px;
        padding: 28px 32px;
        color: #f4fff8;
        box-shadow: 0 12px 35px rgba(20, 76, 52, .16);
        margin-bottom: 22px;
      }
      .hero-kicker { color: #d0f0dc !important; font-size: 12px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; }
      .hero h1 { color: #ffffff !important; font-family: 'Manrope', sans-serif; font-size: clamp(30px, 4vw, 43px); margin: 7px 0 4px 0; font-weight: 800; }
      .hero p { color: #f0faf3 !important; margin: 0; font-size: 15px; max-width: 850px; }
      .section-heading { font-family: 'Manrope', sans-serif; color: #173c2c !important; font-weight: 800; font-size: 21px; margin: 4px 0 2px 0; }
      .muted { color: #53665b !important; font-size: 13px; }
      .result-card { background: linear-gradient(145deg, #ffffff, #f5fbf7); border: 1px solid #dcebe1; border-radius: 18px; padding: 20px 22px; height: 100%; }
      .result-label { color: #53665b !important; text-transform: uppercase; letter-spacing: 1.1px; font-size: 11px; font-weight: 800; }
      .result-class { color: #155b3e !important; font-family: 'Manrope', sans-serif; font-size: 27px; font-weight: 800; line-height: 1.2; margin-top: 7px; }
      .pill { display: inline-block; background: #e5f5eb; color: #145b3d !important; border-radius: 100px; padding: 5px 11px; font-size: 12px; font-weight: 800; margin-top: 11px; }
      .info-card { background: #ffffff; border: 1px solid #dbe6df; border-radius: 16px; padding: 17px 18px; height: 100%; }
      .info-card h4 { color: #1b4935 !important; margin: 0 0 7px 0; font-family: 'Manrope', sans-serif; }
      .info-card p { color: #4e6256 !important; font-size: 13px; margin: 0; }
      .small-note { color: #53665b !important; font-size: 12px; }
      hr { border-color: #d9e4dc; }

      /* Alert text remains dark and readable. */
      [data-testid="stAlert"] p, [data-testid="stAlert"] span { color: #24392d !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model selection: use the stronger lightweight model from held-out evaluation.
# -----------------------------------------------------------------------------
CHECKPOINT_PATH = ARTIFACT_DIR / "student_ce_best.pth"

if not CHECKPOINT_PATH.exists():
    st.error(
        f"The selected checkpoint was not found: {CHECKPOINT_PATH}. "
        "Check that the trained model files are inside the artifacts folder."
    )
    st.stop()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading the trained waste classifier...")
def get_model(path: str, device_name: str):
    return load_student(path, torch.device(device_name))


model = get_model(str(CHECKPOINT_PATH), str(device))
model.eval()
model_image_size = getattr(model, "image_size", IMAGE_SIZE)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-family:Manrope;font-size:25px;font-weight:800;'>♻️ WasteWise AI</div>"
        "<div style='color:#b8d6c5;font-size:12px;margin-top:3px;'>SMART MATERIAL CLASSIFICATION</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**ACTIVE MODEL**")
    st.write("Lightweight Waste CNN · Student CE")
    st.caption("Selected using held-out test performance among the two student models.")
    st.markdown("---")
    st.markdown("**RUNTIME**")
    st.write(f"Device: `{str(device).upper()}`")
    st.write(f"Input: `{model_image_size} × {model_image_size}`")
    st.write(f"Classes: `{len(CLASS_NAMES)}`")
    st.markdown("---")
    st.markdown("**HELD-OUT TEST BENCHMARK**")
    st.metric("Accuracy", "72.65%")
    st.metric("Macro F1", "72.12%")
    st.metric("Checkpoint size", "1.83 MB")
    st.caption("Benchmark recorded on 713 test images. Batch throughput and live single-image response time may differ.")


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Computer vision · Edge AI research demo</div>
      <h1>See waste differently.</h1>
      <p>Classify waste images into nine material categories, inspect the model's confidence,
      and visualise which image regions influenced its prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


classify_tab, performance_tab, about_tab = st.tabs(
    ["🔎  Classify waste", "📊  Model performance", "ℹ️  About the system"]
)


# -----------------------------------------------------------------------------
# Tab 1: image analysis
# -----------------------------------------------------------------------------
with classify_tab:
    st.markdown('<div class="section-heading">Image analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Upload one or more images, or capture an image using your camera.</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose waste image files",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        help="You can select multiple images for batch classification.",
        key="waste_image_uploader",
    )
    camera_capture = st.camera_input("Or take a picture", key="waste_camera_input")

    sources: list[tuple[str, bytes]] = []
    for file in uploaded_files or []:
        sources.append((file.name, file.getvalue()))
    if camera_capture is not None:
        sources.append(("Camera capture.jpg", camera_capture.getvalue()))

    action_col, count_col = st.columns([1, 3])
    with action_col:
        analyze_clicked = st.button(
            "Analyze image(s)",
            type="primary",
            use_container_width=True,
            disabled=len(sources) == 0,
        )
    with count_col:
        st.markdown(
            f'<div style="padding:10px 2px;color:#63736b;font-size:13px;">'
            f'{len(sources)} image(s) ready for analysis · Processing runs locally on this device.</div>',
            unsafe_allow_html=True,
        )

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    if analyze_clicked:
        new_results = []
        progress = st.progress(0, text="Preparing image analysis...")
        errors = []

        for index, (filename, raw_bytes) in enumerate(sources):
            try:
                image = Image.open(BytesIO(raw_bytes)).convert("RGB")
                start_time = time.perf_counter()
                predictions, tensor = predict_image(
                    model,
                    image,
                    device,
                    image_size=model_image_size,
                    top_k=len(CLASS_NAMES),
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if not predictions:
                    raise RuntimeError("The model returned no predictions.")

                result = {
                    "filename": filename,
                    "display_name": f"{index + 1}. {filename}",
                    "image": image.copy(),
                    "predictions": predictions,
                    "tensor": tensor.detach().cpu(),
                    "elapsed_ms": elapsed_ms,
                }
                new_results.append(result)
                st.session_state.prediction_history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image": filename,
                        "prediction": predictions[0]["class"],
                        "confidence": predictions[0]["confidence"],
                    },
                )
            except Exception as error:
                errors.append(f"{filename}: {error}")

            progress.progress(
                (index + 1) / max(len(sources), 1),
                text=f"Analysed {index + 1} of {len(sources)} image(s)...",
            )

        progress.empty()
        st.session_state.analysis_results = new_results
        st.session_state.pop("selected_result_index", None)

        if errors:
            for message in errors:
                st.warning(f"Could not process {message}")
        if new_results:
            st.success(f"Completed classification for {len(new_results)} image(s).")
        elif not errors:
            st.warning("No images were processed.")

    results = st.session_state.analysis_results

    if results:
        st.markdown("---")
        st.markdown('<div class="section-heading">Analysis results</div>', unsafe_allow_html=True)

        selected_index = st.selectbox(
            "Select an analysed image",
            options=list(range(len(results))),
            format_func=lambda index: results[index]["display_name"],
            key="selected_result_index",
        )
        selected = results[selected_index]
        predictions = selected["predictions"]
        best = predictions[0]
        second_confidence = predictions[1]["confidence"] if len(predictions) > 1 else 0.0
        confidence_margin = (best["confidence"] - second_confidence) * 100

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Predicted category", best["class"])
        metric2.metric("Model confidence", f"{best['confidence'] * 100:.2f}%")
        metric3.metric("Top-two margin", f"{confidence_margin:.2f} pp")

        if best["confidence"] < 0.50:
            st.warning(
                "The model's top class has confidence below 50%. Treat this prediction as uncertain "
                "and inspect the other class scores and image quality."
            )
        elif confidence_margin < 10:
            st.info("The top two class scores are close. The model may be uncertain between these categories.")

        image_col, chart_col = st.columns([1, 1.15], gap="large")
        with image_col:
            st.markdown("**Input image**")
            st.image(selected["image"], use_container_width=True)
            st.caption(
                f"{selected['filename']} · {selected['image'].width} × {selected['image'].height} pixels"
            )
        with chart_col:
            st.markdown("**Class probability distribution**")
            chart_data = pd.DataFrame(
                {
                    "Category": [item["class"] for item in predictions],
                    "Score": [float(item["confidence"]) * 100 for item in predictions],
                }
            ).sort_values("Score", ascending=True)

            fig, ax = plt.subplots(figsize=(8.2, 5.0))
            bar_colors = [
                "#159866" if category == best["class"] else "#9bb8a8"
                for category in chart_data["Category"]
            ]
            bars = ax.barh(
                chart_data["Category"],
                chart_data["Score"],
                color=bar_colors,
                height=0.68,
            )
            ax.set_xlim(0, 100)
            ax.set_xlabel("Model score (%)", color="#5d7065", fontsize=10)
            ax.set_ylabel("")
            ax.set_title("Scores across all nine categories", loc="left", color="#173c2c", weight="bold", pad=12)
            ax.grid(axis="x", alpha=0.18, linestyle="--")
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#dce7df")
            ax.tick_params(axis="both", labelsize=9, colors="#53645a", length=0)
            for bar, score in zip(bars, chart_data["Score"]):
                ax.text(
                    min(score + 1.0, 95),
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#34483c",
                )
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "These are softmax model scores, not calibrated guarantees that the prediction is correct."
            )

        explain_col, runtime_col = st.columns(2)
        with explain_col:
            st.metric("Image processing time", f"{selected['elapsed_ms']:.1f} ms")
        with runtime_col:
            st.metric("Inference device", str(device).upper())

        explain_tab, batch_tab, history_tab = st.tabs(
            ["🧠  Grad-CAM explanation", "🗂️  Batch summary", "🕘  Session history"]
        )

        with explain_tab:
            st.markdown("**Which image regions influenced the prediction?**")
            try:
                heatmap = gradcam(
                    model,
                    selected["tensor"].to(device),
                    class_index=None,
                )
                heatmap = np.asarray(heatmap, dtype=np.float32)
                heatmap = np.nan_to_num(heatmap)
                heatmap = np.clip(heatmap, 0.0, 1.0)
                heatmap_rgb = (plt.get_cmap("turbo")(heatmap)[..., :3] * 255).astype(np.uint8)
                heatmap_image = Image.fromarray(heatmap_rgb).resize(selected["image"].size)
                overlay = Image.blend(selected["image"].convert("RGB"), heatmap_image, alpha=0.40)

                original_col, overlay_col = st.columns(2)
                with original_col:
                    st.image(selected["image"], caption="Original image", use_container_width=True)
                with overlay_col:
                    st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
                st.caption(
                    "Grad-CAM is a qualitative explanation of model activation. Highlighted regions do not "
                    "prove that the model used a valid or causally meaningful feature."
                )
            except Exception as error:
                st.warning(f"Grad-CAM could not be generated for this image: {error}")

        with batch_tab:
            batch_rows = []
            for result in results:
                top = result["predictions"][0]
                batch_rows.append(
                    {
                        "Image": result["filename"],
                        "Predicted category": top["class"],
                        "Confidence (%)": round(float(top["confidence"]) * 100, 2),
                        "Processing time (ms)": round(result["elapsed_ms"], 1),
                    }
                )
            st.dataframe(pd.DataFrame(batch_rows), use_container_width=True, hide_index=True)
            if len(batch_rows) > 1:
                st.download_button(
                    "Download batch results (CSV)",
                    data=pd.DataFrame(batch_rows).to_csv(index=False).encode("utf-8"),
                    file_name="wastewise_batch_predictions.csv",
                    mime="text/csv",
                )

        with history_tab:
            history = st.session_state.prediction_history[:50]
            if history:
                history_frame = pd.DataFrame(history)
                history_frame["confidence"] = (history_frame["confidence"] * 100).round(2)
                history_frame = history_frame.rename(
                    columns={"time": "Time", "image": "Image", "prediction": "Prediction", "confidence": "Confidence (%)"}
                )
                st.dataframe(history_frame, use_container_width=True, hide_index=True)
            else:
                st.info("Your prediction history will appear here after the first analysis.")
    else:
        st.markdown(
            """
            <div class="info-card" style="margin-top:12px;">
              <h4>Ready for your first classification</h4>
              <p>Add an image above and select <b>Analyze image(s)</b>. The model will return a category,
              confidence scores for all nine classes, and an optional Grad-CAM visual explanation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Tab 2: held-out model comparison and saved evaluation outputs
# -----------------------------------------------------------------------------
with performance_tab:
    st.markdown('<div class="section-heading">Held-out test performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Measured on the same 713-image test partition. These are experimental results, not live app predictions.</div>',
        unsafe_allow_html=True,
    )

    comparison = pd.DataFrame(
        [
            {"Model": "ResNet-50 teacher", "Accuracy": 0.899018, "Macro F1": 0.903421, "Size (MB)": 90.0500, "Parameters": 23526473, "Latency (ms/image)": 44.8818, "FPS": 22.2808},
            {"Model": "Student CE", "Accuracy": 0.726508, "Macro F1": 0.721217, "Size (MB)": 1.8322, "Parameters": 458667, "Latency (ms/image)": 11.5811, "FPS": 86.3478},
            {"Model": "Student KD", "Accuracy": 0.715288, "Macro F1": 0.711479, "Size (MB)": 1.8322, "Parameters": 458667, "Latency (ms/image)": 11.4731, "FPS": 87.1603},
        ]
    )

    selected_metrics = st.columns(4)
    selected_metrics[0].metric("Selected model accuracy", "72.65%")
    selected_metrics[1].metric("Selected model macro-F1", "72.12%")
    selected_metrics[2].metric("Parameters", "458,667")
    selected_metrics[3].metric("Checkpoint size", "1.83 MB")

    st.markdown("**Teacher vs. student comparison**")
    accuracy_fig, accuracy_ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(comparison))
    width = 0.34
    accuracy_ax.bar(x - width / 2, comparison["Accuracy"] * 100, width, label="Accuracy", color="#247b56")
    accuracy_ax.bar(x + width / 2, comparison["Macro F1"] * 100, width, label="Macro F1", color="#9ab9a8")
    accuracy_ax.set_xticks(x, comparison["Model"])
    accuracy_ax.set_ylabel("Score (%)")
    accuracy_ax.set_ylim(0, 100)
    accuracy_ax.grid(axis="y", alpha=0.2, linestyle="--")
    accuracy_ax.set_axisbelow(True)
    accuracy_ax.spines[["top", "right"]].set_visible(False)
    accuracy_ax.legend(frameon=False, ncol=2)
    accuracy_fig.tight_layout()
    st.pyplot(accuracy_fig, use_container_width=True)
    plt.close(accuracy_fig)

    st.dataframe(
        comparison.assign(
            Accuracy=(comparison["Accuracy"] * 100).round(2).astype(str) + "%",
            **{"Macro F1": (comparison["Macro F1"] * 100).round(2).astype(str) + "%"},
            **{"Size (MB)": comparison["Size (MB)"].round(2)},
            **{"Latency (ms/image)": comparison["Latency (ms/image)"].round(2)},
            FPS=comparison["FPS"].round(2),
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The Student CE model is used by this application because it outperformed Student KD on the held-out test set. "
        "In this experiment, KD did not improve student accuracy: 71.53% vs. 72.65% for Student CE."
    )

    st.markdown("**Saved confusion matrices**")
    confusion_columns = st.columns(3)
    confusion_files = [
        ("Teacher", ARTIFACT_DIR / "confusion_teacher_resnet50.png"),
        ("Student CE", ARTIFACT_DIR / "confusion_student_ce.png"),
        ("Student KD", ARTIFACT_DIR / "confusion_student_kd.png"),
    ]
    for column, (label, path) in zip(confusion_columns, confusion_files):
        with column:
            st.markdown(f"**{label}**")
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.caption("Confusion matrix not found. Run the saved-model evaluation script to generate it.")

    report_path = ARTIFACT_DIR / "classification_report_student_ce.csv"
    if report_path.exists():
        with st.expander("Student CE per-class classification report"):
            try:
                report_df = pd.read_csv(report_path, index_col=0)
                st.dataframe(report_df, use_container_width=True)
            except Exception as error:
                st.warning(f"Could not read the classification report: {error}")

    csv_path = ARTIFACT_DIR / "comparison.csv"
    if csv_path.exists():
        st.download_button(
            "Download measured comparison CSV",
            data=csv_path.read_bytes(),
            file_name="waste_model_comparison.csv",
            mime="text/csv",
        )


# -----------------------------------------------------------------------------
# Tab 3: methodology and model notes
# -----------------------------------------------------------------------------
with about_tab:
    st.markdown('<div class="section-heading">How WasteWise AI works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">A compact computer-vision pipeline developed for waste material classification.</div>',
        unsafe_allow_html=True,
    )

    step_cols = st.columns(4)
    steps = [
        ("01 · INPUT", "Image capture", "Upload an image or use the camera input to provide a waste sample."),
        ("02 · PREPROCESS", "Image preparation", f"Convert to RGB, resize/crop and normalise to the model's {model_image_size} × {model_image_size} input."),
        ("03 · INFERENCE", "Lightweight CNN", "The trained student network extracts visual features and generates scores for nine classes."),
        ("04 · OUTPUT", "Explain the result", "Display the predicted class, class scores, processing time and Grad-CAM visualisation."),
    ]
    for column, (kicker, title, description) in zip(step_cols, steps):
        with column:
            st.markdown(
                f"<div class='info-card'><div class='result-label'>{kicker}</div><h4 style='margin-top:8px'>{title}</h4><p>{description}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Classification categories")
    category_columns = st.columns(3)
    for index, category in enumerate(CLASS_NAMES):
        with category_columns[index % 3]:
            st.markdown(
                f"<div style='background:#ffffff;border:1px solid #e1eae4;border-radius:10px;padding:10px 13px;margin:5px 0;color:#244735;font-weight:600;'>"
                f"<span style='color:#70a68a;margin-right:8px;'>●</span>{category}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Experimental notes")
    st.write(
        "The system was trained on the UCI RealWaste image dataset. A ResNet-50 teacher was compared "
        "with a lightweight student trained using ordinary cross-entropy (CE) and a student trained "
        "using knowledge distillation (KD). The current interface uses Student CE because it achieved "
        "the better held-out result among the two lightweight students."
    )
    st.caption(
        "Important limitation: predictions depend on image quality, lighting, object visibility and "
        "similarity to the training dataset. This is a research demonstration, not an automated waste "
        "disposal or recycling authority."
    )


st.markdown("---")
footer_left, footer_right = st.columns([3, 1])
with footer_left:
    st.markdown(
        '<div class="small-note">WasteWise AI · Lightweight CNN waste classification research prototype</div>',
        unsafe_allow_html=True,
    )
with footer_right:
    st.markdown(
        f'<div class="small-note" style="text-align:right;">Local inference · {str(device).upper()}</div>',
        unsafe_allow_html=True,
    )
