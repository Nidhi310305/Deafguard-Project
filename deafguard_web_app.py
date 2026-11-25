from pathlib import Path
from datetime import datetime
import html

import streamlit as st

from deafguard_audio_predictor import predict_uploaded_wav
from deafguard_response import create_alert_response
from deafguard_vibration import trigger_vibration


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="DeafGuard | Major Project Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Project data
# -----------------------------------------------------------------------------

SOUND_EVENTS = [
    "emergency_siren",
    "vehicle_horn",
    "glass_breaking",
    "baby_crying",
    "door_knocking",
    "footsteps",
    "other_unknown",
]

ISL_OUTPUTS = {
    "doctor": "I need medical assistance.",
    "emergency": "This is an emergency. Please help me.",
    "fall": "I may have fallen. Please check on me.",
    "food": "I need food.",
    "hello": "Hello.",
    "help": "I need help.",
    "no": "No.",
    "pain": "I am in pain. Please help me.",
    "safe": "I am safe.",
    "stop": "Please stop.",
    "thank_you": "Thank you.",
    "tired": "I am tired.",
    "unsafe": "I feel unsafe. Please help me.",
    "water": "I need water.",
    "yes": "Yes.",
}

ICON_FILES = {
    event: f"{event}.png"
    for event in SOUND_EVENTS + list(ISL_OUTPUTS.keys())
}

SEVERITY_COLORS = {
    "CRITICAL": "#b91c1c",
    "HIGH": "#c2410c",
    "MEDIUM": "#1d4ed8",
    "LOW": "#0f766e",
    "NONE": "#334155",
}


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --navy: #0b1220;
        --panel: #121d2f;
        --panel2: #17243a;
        --gold: #f5b400;
        --muted: #9fb0c8;
        --line: #2b3b55;
        --teal: #14b8a6;
    }

    .stApp {
        background: linear-gradient(135deg, #07101d 0%, #0b1526 55%, #101d31 100%);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem 1.8rem;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: linear-gradient(115deg, #101c31, #152a43);
        box-shadow: 0 15px 45px rgba(0,0,0,.20);
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin: 0;
        color: var(--gold);
        font-size: 3rem;
        letter-spacing: -.04em;
    }

    .hero p {
        margin: .35rem 0 0 0;
        color: #dbe7f5;
        font-size: 1.05rem;
    }

    .section-title {
        color: #eef5ff;
        font-size: 1.35rem;
        font-weight: 750;
        margin: 1rem 0 .7rem 0;
    }

    .status-card {
        padding: 1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(18, 29, 47, .94);
        min-height: 105px;
    }

    .status-label {
        color: var(--muted);
        text-transform: uppercase;
        font-size: .72rem;
        letter-spacing: .08em;
        font-weight: 700;
    }

    .status-value {
        color: #f8fbff;
        font-size: 1.12rem;
        font-weight: 700;
        margin-top: .35rem;
    }

    .status-good { color: #34d399; }
    .status-warn { color: #fbbf24; }
    .status-blue { color: #60a5fa; }

    .panel {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.15rem;
        background: rgba(18, 29, 47, .82);
        min-height: 100%;
    }

    .alert-box {
        padding: 1rem 1.15rem;
        border-radius: 15px;
        color: white;
        margin: .8rem 0;
        border-left: 7px solid rgba(255,255,255,.75);
    }

    .alert-box h2 {
        margin: 0 0 .35rem 0;
        font-size: 1.35rem;
    }

    .alert-box p { margin: .2rem 0; }

    .metric-note {
        color: var(--muted);
        font-size: .82rem;
        margin-top: -.45rem;
    }

    .loop-step {
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: .75rem .9rem;
        background: #15243a;
        color: #e7effa;
        margin: .35rem 0;
    }

    .loop-arrow {
        text-align: center;
        color: var(--gold);
        font-weight: 800;
    }

    .small-muted {
        color: var(--muted);
        font-size: .82rem;
    }

    [data-testid="stMetricValue"] { color: #f8fbff; }
    [data-testid="stMetricLabel"] { color: #9fb0c8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def humanize(value: str) -> str:
    return value.replace("_", " ").replace("+", " + ").title()


def icon_path(key: str):
    """Find an icon in common project asset locations."""
    filename = ICON_FILES.get(key, "other_unknown.png")
    candidates = [
        Path("assets") / filename,
        Path("symbols") / filename,
        Path("icons") / filename,
        Path(filename),
        Path("deafguard_icons") / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def display_icon(key: str, width: int = 180):
    path = icon_path(key)
    if path:
        st.image(path, width=width)
    else:
        st.markdown(
            f"<div style='font-size:6rem;text-align:center'>⚠️</div>",
            unsafe_allow_html=True,
        )
        st.caption("Copy the generated PNG icons into an assets/ folder to display them here.")


def render_status_card(label: str, value: str, color_class: str = ""):
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{html.escape(label)}</div>
            <div class="status-value {color_class}">{html.escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert(response, prediction=None, filename=""):
    severity = getattr(response, "severity", "NONE")
    event = getattr(response, "event", "other_unknown")
    message = getattr(response, "message", "No message available.")
    environment = getattr(response, "environment", "uncertain")
    vibration = getattr(response, "vibration_pattern", "none")
    priority = getattr(response, "priority", 0)
    confidence = getattr(response, "confidence", 0.0)
    color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["NONE"])

    left, right = st.columns([1, 2])
    with left:
        display_icon(event, width=165)
        st.markdown(f"### {humanize(event)}")
        st.caption(f"Environment: {humanize(environment)}")

    with right:
        st.markdown(
            f"""
            <div class="alert-box" style="background:{color}">
                <h2>{html.escape(severity)} ALERT · {html.escape(humanize(event).upper())}</h2>
                <p>{html.escape(message)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        a, b, c = st.columns(3)
        a.metric("Confidence", f"{confidence:.1%}")
        b.metric("Priority", str(priority))
        c.metric("Vibration", humanize(vibration))

        if prediction:
            st.caption(f"Source audio: {filename}")
            with st.expander("Top model predictions"):
                for item in prediction.get("top_predictions", []):
                    st.write(
                        f"{humanize(item['label'])}: "
                        f"{item['confidence']:.2%}"
                    )


def add_history(kind: str, label: str, confidence: float, action: str):
    record = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": kind,
        "event": humanize(label),
        "confidence": f"{confidence:.1%}",
        "action": action,
    }
    st.session_state.setdefault("history", [])
    st.session_state["history"].insert(0, record)
    st.session_state["history"] = st.session_state["history"][:10]


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

for key, default in {
    "sound_response": None,
    "sound_prediction": None,
    "sound_filename": "",
    "isl_label": None,
    "isl_text": None,
    "history": [],
}.items():
    st.session_state.setdefault(key, default)


# -----------------------------------------------------------------------------
# Header and sidebar
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>DeafGuard</h1>
        <p><b>Safety-Aware Multimodal Accessibility System</b></p>
        <p>Environmental sound awareness, Indian Sign Language communication,
        haptic feedback, speech output, and confirmation-gated assistance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Project control")
    st.info("This dashboard is the research-prototype interface. The real-time microphone monitor and Android companion remain separate runtime components.")
    st.markdown("### Current validated results")
    st.metric("Sound accuracy", "92.9%")
    st.caption("56 untouched test clips · 7 classes")
    st.metric("ISL accuracy", "96.0%")
    st.caption("126 holdout samples · 15 gestures")
    st.markdown("### Decision policy")
    st.write("• Confidence threshold: 0.50")
    st.write("• Temporal validation: 3 windows")
    st.write("• Duplicate cooldown: 5 seconds")
    st.write("• SOS: explicit confirmation required")

st.markdown('<div class="section-title">System status</div>', unsafe_allow_html=True)
status_cols = st.columns(4)
with status_cols[0]:
    render_status_card("Sound classifier", "Ready", "status-good")
with status_cols[1]:
    render_status_card("ISL classifier", "15 gestures", "status-blue")
with status_cols[2]:
    render_status_card("Alert decision layer", "Protected", "status-good")
with status_cols[3]:
    render_status_card("Android companion", "Use live monitor", "status-warn")


# -----------------------------------------------------------------------------
# Main dashboard tabs
# -----------------------------------------------------------------------------

tab_overview, tab_sound, tab_isl, tab_history = st.tabs(
    ["Overview", "Sound awareness", "ISL communication", "Alert history"]
)

with tab_overview:
    st.markdown('<div class="section-title">Validated project performance</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sound accuracy", "92.9%", "56 test clips")
    metric_cols[1].metric("ISL accuracy", "96.0%", "126 holdout samples")
    metric_cols[2].metric("Sound classes", "7", "YAMNet embeddings")
    metric_cols[3].metric("ISL gestures", "15", "Landmark classifier")

    st.markdown('<div class="section-title">Safety-aware multimodal response loop</div>', unsafe_allow_html=True)
    loop_cols = st.columns(3)
    loop_steps = [
        ("1", "Sense", "Microphone or camera receives environmental input."),
        ("2", "Classify", "YAMNet features or hand landmarks produce a prediction."),
        ("3", "Check", "Confidence and temporal stability are evaluated."),
        ("4", "Alert", "Visual symbol, severity, vibration, and message are produced."),
        ("5", "Communicate", "ISL becomes understandable English and speech."),
        ("6", "Assist", "SOS is offered only after a safety-related sign and confirmation."),
    ]
    for index, (number, title, description) in enumerate(loop_steps):
        with loop_cols[index % 3]:
            st.markdown(
                f"<div class='loop-step'><b>{number}. {title}</b><br><span class='small-muted'>{description}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Research notes</div>', unsafe_allow_html=True)
    st.warning(
        "The 96.0% ISL result is a preliminary holdout result from a limited signer dataset. "
        "The fall gesture had 66.7% recall and requires additional cross-user validation."
    )

with tab_sound:
    left, right = st.columns([1.05, 1.4])

    with left:
        st.markdown('<div class="section-title">Environmental sound awareness</div>', unsafe_allow_html=True)
        st.write("Upload an external WAV clip to run the trained seven-class sound pipeline.")
        uploaded_audio = st.file_uploader(
            "WAV sound recording",
            type=["wav"],
            key="sound_upload",
            help="The trained pipeline converts the clip to 16 kHz mono, extracts YAMNet embeddings, and applies the DeafGuard classifier.",
        )

        if uploaded_audio is not None:
            st.audio(uploaded_audio, format="audio/wav")
            if st.button("Run DeafGuard sound prediction", type="primary", use_container_width=True):
                try:
                    with st.spinner("Running YAMNet and the trained classifier..."):
                        prediction = predict_uploaded_wav(uploaded_audio)
                        response = create_alert_response(
                            prediction["event"],
                            prediction["confidence"],
                        )
                    st.session_state["sound_prediction"] = prediction
                    st.session_state["sound_response"] = response
                    st.session_state["sound_filename"] = uploaded_audio.name
                    add_history(
                        "Sound",
                        prediction["event"],
                        prediction["confidence"],
                        f"{response.severity} response generated",
                    )
                    st.success("Sound prediction completed.")
                except Exception as error:
                    st.error(f"Could not process this WAV file: {error}")

        st.markdown("#### Decision-layer policy")
        policy_cols = st.columns(3)
        policy_cols[0].metric("Threshold", "0.50")
        policy_cols[1].metric("Validation", "3 windows")
        policy_cols[2].metric("Cooldown", "5 sec")
        st.caption("The Streamlit upload view demonstrates classification. Use the real-time sound monitor for microphone inference and temporal alert validation.")

    with right:
        st.markdown('<div class="section-title">Current sound response</div>', unsafe_allow_html=True)
        if st.session_state["sound_response"] is not None:
            render_alert(
                st.session_state["sound_response"],
                st.session_state["sound_prediction"],
                st.session_state["sound_filename"],
            )
            if st.button("Test software vibration pattern", key="dashboard_vibrate"):
                try:
                    result = trigger_vibration(
                        st.session_state["sound_response"].vibration_pattern
                    )
                    st.success(
                        f"{result['status']} — {result['pulses']} pulse(s)"
                    )
                except Exception as error:
                    st.error(f"Vibration test failed: {error}")
        else:
            st.info("No uploaded sound has been processed yet. The live microphone monitor and Android companion can still run separately.")

with tab_isl:
    left, right = st.columns([1.0, 1.35])

    with left:
        st.markdown('<div class="section-title">15-gesture ISL communication</div>', unsafe_allow_html=True)
        isl_gesture = st.selectbox(
            "Recognized gesture or stable sequence",
            list(ISL_OUTPUTS.keys()),
            format_func=humanize,
            key="isl_selector",
        )
        isl_confidence = st.slider(
            "Demonstration confidence",
            0.0,
            1.0,
            0.96,
            0.01,
            key="isl_confidence",
        )
        if st.button("Process ISL response", type="primary", use_container_width=True):
            st.session_state["isl_label"] = isl_gesture
            st.session_state["isl_text"] = ISL_OUTPUTS[isl_gesture]
            add_history(
                "ISL",
                isl_gesture,
                isl_confidence,
                "English message generated",
            )
            st.success("ISL response processed.")

        st.markdown("#### SOS policy")
        st.write("Direct safety candidates: `help`, `emergency`, `unsafe`, and `fall`.")
        st.write("Medical-assistance candidates: `pain` and `doctor`.")
        st.caption("In the live application, an SOS must remain confirmation-gated. The demonstration selector does not send an external message.")

    with right:
        st.markdown('<div class="section-title">Communication output</div>', unsafe_allow_html=True)
        if st.session_state["isl_label"]:
            label = st.session_state["isl_label"]
            display_icon(label, width=155)
            st.success("Stable ISL sign accepted")
            st.markdown(f"### {ISL_OUTPUTS[label]}")
            st.write(f"Recognized gesture: **{humanize(label)}**")
            st.write(f"Confidence: **{isl_confidence:.1%}**")
            st.write("Speech output: **Ready**")
            if label in {"help", "emergency", "unsafe", "fall", "pain", "doctor"}:
                st.warning("This gesture belongs to a safety or medical-assistance category. The live Android app should request confirmation before SOS.")
        else:
            st.info("Select a gesture and process it to preview the communication output.")

with tab_history:
    st.markdown('<div class="section-title">Recent multimodal events</div>', unsafe_allow_html=True)
    if st.session_state["history"]:
        st.dataframe(
            st.session_state["history"],
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear dashboard history"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("No events recorded in this dashboard session yet.")

    st.markdown('<div class="section-title">Deployment note</div>', unsafe_allow_html=True)
    st.write(
        "The Streamlit dashboard is the visual research interface. The real-time microphone monitor, "
        "alert server, and Android companion are the operational path for live sound alerts, vibration, "
        "automatic speech, and confirmation-gated SOS behavior."
    )
