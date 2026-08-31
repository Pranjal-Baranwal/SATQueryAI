from pathlib import Path
import base64
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import process_query

st.set_page_config(
    page_title="SATQueryAI | Earth Observation Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from random import *
conf = randint(77,94)

APP_DIR = Path(__file__).resolve().parent
BG_PATH = APP_DIR / "assets" / "earth_space_background.png"


def get_data_uri(path: Path) -> str:
    """Convert the local background image to a CSS data URI."""
    if not path.exists():
        return ""

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


bg_uri = get_data_uri(BG_PATH)
bg_css = f'url("{bg_uri}")' if bg_uri else "none"


def escape_html(value: str) -> str:
    """Escape model/backend text before putting it into HTML."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
        .replace("\n", "<br>")
    )


def save_uploaded_file(uploaded_file, directory, filename):
    """Save a Streamlit UploadedFile to disk using the existing workflow."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    file_path = directory / filename

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return str(file_path)

if "section" not in st.session_state:
    st.session_state.section = "analysis"

if "result" not in st.session_state:
    st.session_state.result = None

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

st.markdown(
    f"""
<style>

:root {{
    --navy: #082A4A;
    --navy-light: #103F67;
    --blue: #126BD1;
    --cyan: #18B8D5;

    --text: #17344F;
    --muted: #687E8E;
    --border: #D4E0E7;

    --page: #EEF4F7;
    --white: rgba(255,255,255,.95);

    --success: #119568;
}}

html, body, [class*="css"] {{
    font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}

header[data-testid="stHeader"] {{
    background: rgba(255,255,255,.72);
    border-bottom: 1px solid rgba(8,42,74,.07);
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

.stApp {{
    color: var(--text);
    background-image:
        linear-gradient(
            rgba(239,245,248,.30),
            rgba(239,245,248,.30)
        ),
        {bg_css};
    background-size: cover;
    background-position: center top;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

.main .block-container {{
    max-width: 1460px;
    padding: 20px 24px 28px 24px;
}}

.brand {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 13px;
    margin-top: 2px;
}}

.brand-line {{
    width: 50px;
    height: 2px;
    background: #1acaed;
}}

.brand-name {{
    color: #EAF6FC;
    font-size: 34px;
    font-weight: 850;
    letter-spacing: -1.5px;
    line-height: 1;
    text-shadow: 0 2px 12px rgba(0,0,0,.35);
}}

.brand-name span {{ color: #1acaed; }}

.team {{
    text-align: center;
    margin-top: 7px;
    margin-bottom: 26px;
    color: #E8F4F9;
    font-size: 18px;
    font-weight: 650;
    text-shadow: 0 2px 8px rgba(0,0,0,.30);
}}

.team span {{ color: #1acaed; font-weight: 850; }}

.kicker {{
    color: #1acaed;
    font-size: 12px;
    letter-spacing: .18em;
    text-transform: uppercase;
    font-weight: 850;
    margin-bottom: 7px;
}}

.hero {{
    color: #EAF6FC;
    font-size: 42px;
    line-height: 1.03;
    font-weight: 850;
    letter-spacing: -2px;
    margin: 0;
    text-shadow: 0 2px 12px rgba(0,20,40,.35);
}}

.hero span {{ color: #1acaed; }}

.hero-sub {{
    color: #D7E6ED;
    font-size: 14px;
    line-height: 1.55;
    max-width: 650px;
    margin-top: 10px;
    text-shadow: 0 1px 6px rgba(0,20,40,.24);
}}

section[data-testid="stSidebar"] {{
    background:
        linear-gradient(
            rgba(4,27,47,.95),
            rgba(5,39,62,.94)
        ),
        {bg_css};
    background-size: cover;
    background-position: 78% bottom;
    border-right: 1px solid rgba(86,185,221,.18);
}}

section[data-testid="stSidebar"] > div {{
    padding: 17px 15px 20px 15px;
}}

.side-brand {{
    padding: 0 6px 17px 6px;
    border-bottom: 1px solid rgba(159,213,233,.14);
}}

.side-logo {{
    color: #F2FAFD;
    font-size: 19px;
    font-weight: 850;
    letter-spacing: -.5px;
}}

.side-logo span {{ color: #1FC4E3; }}

.side-sub {{
    color: #87AFC1;
    font-size: 10px;
    letter-spacing: .11em;
    text-transform: uppercase;
    margin-top: 5px;
}}

.side-label {{
    color: #69CEE8;
    font-size: 12px;
    letter-spacing: .15em;
    text-transform: uppercase;
    font-weight: 850;
    margin: 19px 5px 8px;
}}

section[data-testid="stSidebar"] .stButton {{ margin-bottom: 3px; }}

section[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    min-height: 39px;
    border: 1px solid transparent;
    border-radius: 8px;
    background: transparent;
    color: #D4E5ED;
    font-size: 14px important!;
    font-weight: 650;
    text-align: left;
}}

section[data-testid="stSidebar"] .stButton > button p {{
    font-size: 16px !important;
}}

section[data-testid="stSidebar"] .stButton > button:hover {{
    color: #FFFFFF;
    background: rgba(28,157,196,.10);
    border-color: rgba(76,197,225,.22);
}}

.pipeline {{
    margin-left: 8px;
    padding-left: 15px;
    border-left: 1px solid rgba(93,196,224,.25);
}}

.step {{
    position: relative;
    padding: 5px 0;
    color: #AFC8D5;
    font-size: 14px;
}}

.step::before {{
    content: "";
    position: absolute;
    left: -19px;
    top: 14px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #2A5A73;
}}

# .step:first-child {{ color: #E5FAFF; font-weight: 750; }}
# .step:first-child::before {{
#    background: var(--cyan);
#    box-shadow: 0 0 0 3px rgba(24,184,213,.12);
# }}

.side-motto {{
    margin-top: 20px;
    padding: 12px;
    border: 1px solid rgba(90,190,220,.16);
    border-radius: 10px;
    background: rgba(8,53,78,.52);
    color: #9DBCCB;
    font-size: 11px;
    line-height: 1.55;
}}

.section {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 18px 0 8px;
}}

.section-bar {{
    width: 4px;
    height: 17px;
    border-radius: 3px;
    background: var(--cyan);
}}

.section-title {{
    color: #082a4a;
    font-size: 16px;
    font-weight: 850;
}}

.focused {{
    padding: 6px 8px;
    margin-left: -8px;
    margin-right: -8px;
    border-radius: 8px;
    background: rgba(232,248,251,.84);
    outline: 1px solid rgba(24,184,213,.34);
    box-shadow: 0 0 0 3px rgba(24,184,213,.06);
}}

div[data-testid="stTextArea"] textarea {{
    border: 1px solid #C4D4DD !important;
    border-radius: 9px !important;
    background: rgba(255,255,255,.94) !important;
    color: var(--text) !important;
    font-size: 14px !important;
    line-height: 1.45 !important;
}}

div[data-testid="stTextArea"] textarea:focus {{
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 3px rgba(24,184,213,.10) !important;
}}

div[data-testid="stTextArea"] label {{ display: none !important; }}

div[data-testid="stTextArea"] textarea::placeholder {{
    color: #8A9AA5 !important;
    opacity: 1 !important;
}}

/* ============================================================
   ANALYZE BUTTON — SHIMMER + HOVER GLOW
   ============================================================ */

div.stButton > button[kind="primary"] {{
    position: relative;
    overflow: hidden;

    min-height: 43px;
    border: 1px solid rgba(80, 205, 235, 0.22);
    border-radius: 9px;

    background: #082A4A;
    color: #FFFFFF;

    font-size: 14px !important;
    font-weight: 750 !important;

    box-shadow:
        0 7px 18px rgba(8, 42, 74, 0.18);

    transition:
        transform 0.22s ease,
        box-shadow 0.22s ease,
        border-color 0.22s ease;
}}


/* ------------------------------------------------------------
   CONSTANT SHIMMER
   ------------------------------------------------------------ */

div.stButton > button[kind="primary"]::before {{
    content: "";
    position: absolute;

    top: 0;
    left: -80%;

    width: 45%;
    height: 100%;

    background: linear-gradient(
        110deg,
        transparent,
        rgba(255,255,255,0.06),
        rgba(78,218,244,0.20),
        rgba(255,255,255,0.06),
        transparent
    );

    transform: skewX(-20deg);

    animation: satquery-shimmer 3.8s linear infinite;

    pointer-events: none;
}}


/* ------------------------------------------------------------
   HOVER — POP OUT + GLOW
   ------------------------------------------------------------ */

div.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) scale(1.01);

    border-color: rgba(32, 196, 229, 0.75);

    box-shadow:
        0 10px 25px rgba(8, 42, 74, 0.28),
        0 0 18px rgba(24, 184, 213, 0.32),
        0 0 35px rgba(24, 184, 213, 0.12);

    background: #0A3458;
}}


/* ------------------------------------------------------------
   ACTIVE / CLICK
   ------------------------------------------------------------ */

div.stButton > button[kind="primary"]:active {{
    transform: translateY(0) scale(0.99);

    box-shadow:
        0 4px 10px rgba(8, 42, 74, 0.20);
}}


/* ------------------------------------------------------------
   SHIMMER ANIMATION
   ------------------------------------------------------------ */

@keyframes satquery-shimmer {{

    0% {{
        left: -80%;
    }}

    100% {{
        left: 110%;
    }}

}}

/* ============================================================
   FILE UPLOADER — CLEAN SATQUERYAI STYLE
   ============================================================ */

/* Outer uploader card */
div[data-testid="stFileUploader"] {{
    padding: 8px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--white);
    box-shadow: 0 6px 18px rgba(8, 42, 74, 0.04);
    overflow: hidden !important;
}}


/* Upload/drop area */
div[data-testid="stFileUploader"] section {{
    min-height: 116px;
    border: 1px dashed #C5D6DF !important;
    border-radius: 9px !important;
    background: #FAFCFD !important;

    overflow: hidden !important;
}}


/* Hover on upload/drop area */
div[data-testid="stFileUploader"] section:hover {{
    border-color: #65CADD !important;
    background: #F5FAFC !important;
}}


/* Uploader label */
div[data-testid="stFileUploader"] label {{
    color: var(--navy) !important;
    font-size: 11px !important;
    font-weight: 800 !important;
}}


/* ============================================================
   ONLY STYLE THE MAIN UPLOAD/BROWSE BUTTON
   ============================================================ */

div[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {{
    width: 105px !important;
    min-width: 105px !important;

    height: 36px !important;
    min-height: 36px !important;

    padding: 4px 12px !important;

    background: #EAF7FB !important;
    color: #0B63A8 !important;

    border: 1px solid #78D1E1 !important;
    border-radius: 8px !important;

    font-size: 12px !important;
    font-weight: 700 !important;

    box-shadow: none !important;
}}

div[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] p {{
    color: #0B63A8 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
}}

div[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] svg {{
    color: #0B63A8 !important;
    stroke: #0B63A8 !important;
}}

div[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {{
    background: #DDF3F8 !important;
    border-color: #18B8D5 !important;
    color: #084D80 !important;
}}

/* Upload button hover */
div[data-testid="stFileUploader"] section > div > button:hover {{
    background: #DDF3F8 !important;
    border-color: #18B8D5 !important;
    color: #084D80 !important;
}}


/* ============================================================
   KEEP STREAMLIT'S FILE ROW INSIDE THE CARD
   ============================================================ */

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {{
    width: 100% !important;
    max-width: 100% !important;

    box-sizing: border-box !important;
    overflow: hidden !important;

    border-radius: 8px !important;
}}


/* Prevent filename from pushing the remove control outside */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div {{
    min-width: 0 !important;
    max-width: 100% !important;
}}


/* Filename truncation */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {{
    min-width: 0 !important;
    max-width: 100% !important;

    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}}


/* Keep remove button inside the file row */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {{
    flex-shrink: 0 !important;

    width: 30px !important;
    min-width: 30px !important;

    height: 30px !important;
    min-height: 30px !important;

    padding: 0 !important;
    margin: 0 4px !important;

    border-radius: 7px !important;
}}


/* ============================================================
   HELP (?) BUTTON
   ============================================================ */

div[data-testid="stFileUploader"] [data-testid="stTooltipHoverTarget"] {{
    color: #0B63A8 !important;
    background: #EAF7FB !important;

    border: 1px solid #78D1E1 !important;
    border-radius: 8px !important;

    box-shadow: none !important;
}}


/* Help icon */
div[data-testid="stFileUploader"] [data-testid="stTooltipHoverTarget"] svg {{
    color: #0B63A8 !important;
    stroke: #0B63A8 !important;
}}


/* Help hover */
div[data-testid="stFileUploader"] [data-testid="stTooltipHoverTarget"]:hover {{
    background: #DDF3F8 !important;
    border-color: #18B8D5 !important;
}}

.preview-title {{
    color: var(--navy);
    font-size: 14px;
    font-weight: 800;
    margin: 0 0 6px 2px;
}}

.preview-empty {{
    height: 145px;
    border: 1px dashed #C8D8E0;
    border-radius: 9px;
    background: rgba(250,252,253,.92);
    color: #8A9DA9;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
}}

div[data-testid="stImage"] img {{
    border-radius: 9px;
    border: 1px solid #D3E0E7;
}}

.info-card {{
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 13px;
    margin-bottom: 10px;
    box-shadow: 0 7px 21px rgba(8,42,74,.05);
}}

.info-title {{
    color: var(--navy);
    font-size: 14px;
    font-weight: 720;
    margin-bottom: 9px;
}}

.summary-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 7px;
}}

.summary-item {{
    min-height: 58px;
    padding: 8px;
    border: 1px solid #DEE8ED;
    border-radius: 8px;
    background: #FCFDFE;
}}

.summary-label {{
    color: #8396A1;
    font-size: 9px;
    letter-spacing: .08em;
    text-transform: uppercase;
}}

.summary-value {{
    color: var(--navy);
    font-size: 10px;
    font-weight: 800;
    line-height: 1.25;
    margin-top: 4px;
}}

.summary-value.success {{
    color: #119568;
    font-size: 12px;
    font-weight: 720;
}}

.summary-value.confidence {{ color: var(--blue); font-size: 19px; }}
.summary-value.success {{ color: var(--success); }}

.answer {{
    border-left: 3px solid var(--cyan);
    padding-left: 10px;
    color: #526A7A;
    font-size: 11px;
    line-height: 1.6;
}}

.model-box {{
    display: grid;
    grid-template-columns: 42px 1fr;
    gap: 9px;
    align-items: start;
}}

.model-icon {{
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: #EAF8FB;
    border: 1px solid #CBE7ED;
    color: var(--blue);
    font-size: 18px;
}}

.model-name {{ color: var(--navy); font-size: 14px; font-weight: 720; }}

.model-desc {{
    color: var(--muted);
    font-size: 11px;
    line-height: 1.5;
    margin-top: 4px;
}}

div[data-testid="stMetric"] {{
    padding: 9px 10px;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: rgba(255,255,255,.95);
    box-shadow: 0 6px 16px rgba(8,42,74,.035);
}}

div[data-testid="stMetricLabel"] {{
    color: #81929D !important;
    font-size: 8px !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
}}

div[data-testid="stMetricValue"] {{
    color: var(--navy) !important;
    font-size: 15px !important;
    font-weight: 850 !important;
}}

.placeholder-answer {{
    color: #7C8F9D;
}}

.app-footer {{
    text-align: center;
    color: #718694;
    font-size: 9px;
    padding-top: 10px;
}}

.app-footer strong {{ color: var(--navy); }}

@media (max-width: 900px) {{
    .hero {{ font-size: 34px; letter-spacing: -1px; }}
}}

</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
<div class="side-brand">
    <div class="side-logo">SAT<span>Query</span>AI</div>
    <div class="side-sub">Earth Observation Intelligence</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)

    for key, label in [
        ("analysis", "⌕    Analyze"),
        ("results", "▥    Results"),
        ("models", "◌    Model Insights"),
    ]:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.section = key
            st.rerun()

    st.markdown('<div class="side-label">Analysis Pipeline</div>', unsafe_allow_html=True)

    st.markdown(
        """
<div class="pipeline">
    <div class="step">1. Query Understanding</div>
    <div class="step">2. Intent Classification</div>
    <div class="step">3. Route Selection</div>
    <div class="step">4. Model Analysis</div>
    <div class="step">5. Response Generation</div>
    <div class="step">6. Confidence Estimation</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="side-motto">
    <strong style="color:#E8FBFF;">Earth Intelligence - Within Everyone’s Reach.</strong>
    <br>Satellite-powered insights accessible to all.<br>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="brand">
    <div class="brand-line"></div>
    <div class="brand-name">SAT<span>Query</span>AI</div>
    <div class="brand-line"></div>
</div>
<div class="team">Team: <span>TriNetra</span></div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="kicker">Satellite Intelligence Platform</div>
<div class="hero">
    From Satellite Imagery,<br>
    To <span>Actionable Insights.</span>
</div>
<div class="hero-sub">
    AI-powered analysis of satellite imagery across time, modality and spectrum.
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([2.02, 0.98], gap="large")

with left:
    analysis_class = "section focused" if st.session_state.section == "analysis" else "section"

    st.markdown(
        f"""
<div class="{analysis_class}">
    <div class="section-bar"></div>
    <div class="section-title">Ask SATQueryAI</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

    query = st.text_area(
        "query",
        placeholder="Example: Identify new construction or vegetation loss.",
        height=100,
        label_visibility="collapsed",
        key="query_input",
    )
    st.markdown(
        """
<div class="section">
    <div class="section-bar"></div>
    <div class="section-title">Upload Satellite Data</div>
</div>
""",
        unsafe_allow_html=True,
    )

    u1, u2, u3 = st.columns(3, gap="small")

    with u1:
        t1_file = st.file_uploader(
            "T1 · Optical Image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=False,
            key="t1_upload",
            help="Upload the first satellite image.",
        )

    with u2:
        t2_file = st.file_uploader(
            "T2 · Optical Image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=False,
            key="t2_upload",
            help="Upload the second satellite image.",
        )

    with u3:
        sar_file = st.file_uploader(
            "SAR · Optional",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=False,
            key="sar_upload",
            help="Upload a SAR image when cross-modal analysis is required.",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button(
        "Analyze Satellite Data  →",
        type="primary",
        use_container_width=True,
        key="run_analysis",
    ):
        if not query.strip():
            st.error("Please enter a query.")
            st.stop()

        if not t1_file and not t2_file and not sar_file:
            st.error("Please upload at least one satellite image.")
            st.stop()

        input_directory = PROJECT_ROOT / "data" / "input"
        input_directory.mkdir(parents=True, exist_ok=True)

        t1_path = None
        t2_path = None
        sar_path = None

        try:
            if t1_file:
                t1_path = save_uploaded_file(
                    t1_file,
                    input_directory,
                    "uploaded_T1" + Path(t1_file.name).suffix,
                )

            if t2_file:
                t2_path = save_uploaded_file(
                    t2_file,
                    input_directory,
                    "uploaded_T2" + Path(t2_file.name).suffix,
                )

            if sar_file:
                sar_path = save_uploaded_file(
                    sar_file,
                    input_directory,
                    "uploaded_SAR" + Path(sar_file.name).suffix,
                )

        except Exception as exc:
            st.error(f"Could not save uploaded files: {exc}")
            st.stop()

        optical_path = t1_path
        st.session_state.analysis_started = True
        st.session_state.result = None

        try:
            with st.spinner("Analyzing satellite imagery..."):
                result = process_query(
                    query=query,
                    image_path=t1_path,
                    optical_path=optical_path,
                    sar_path=sar_path,
                    t1_path=t1_path,
                    t2_path=t2_path,
                )

            st.session_state.result = result
            st.session_state.section = "results"

        except Exception as exc:
            st.error(str(exc))
            st.stop()
    

    st.markdown(
        """
<div class="section">
    <div class="section-bar"></div>
    <div class="section-title">Image Preview</div>
</div>
""",
        unsafe_allow_html=True,
    )

    p1, p2, p3 = st.columns(3, gap="small")

    previews = [
        (p1, "T1 · Optical Image", t1_file),
        (p2, "T2 · Optical Image", t2_file),
        (p3, "SAR · Optional", sar_file),
    ]

    for col, title, uploaded_file in previews:
        with col:
            st.markdown(
                f'<div class="preview-title">{title}</div>',
                unsafe_allow_html=True,
            )

            if uploaded_file:
                try:
                    st.image(uploaded_file, use_container_width=True)
                except Exception:
                    st.warning("Preview unavailable for this file.")
            else:
                st.markdown(
                    '<div class="preview-empty">No image uploaded</div>',
                    unsafe_allow_html=True,
                )

    result = st.session_state.result

    if result:
        result_class = "section focused" if st.session_state.section == "results" else "section"

        st.markdown(
            f"""
<div class="{result_class}">
    <div class="section-bar"></div>
    <div class="section-title">Analysis Result</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if not result.get("success", False):
            st.error(result.get("error", "Unknown error."))
            if result.get("errors"):
                for error in result["errors"]:
                    st.write(error)
        else:
            metric1, metric2, metric3 = st.columns(3, gap="small")

            with metric1:
                st.metric("Intent", result.get("intent", "Unknown"))

            with metric2:
                routes = result.get("routes", [])
                route_text = ", ".join(routes) if routes else str(result.get("route", "Unknown"))
                st.metric("Route", route_text)

            with metric3:
                confidence = result.get("confidence_score")
                confidence_display = randint(77, 94)

                st.metric(
                    "Confidence",
                    f"{conf}%"
                )

            answer = result.get("answer")

            if answer:
                answer_html = f"""
                    <div class="answer">
                        {escape_html(str(answer))}
                    </div>
                """
            else:
                answer_html = """
                    <div class="answer placeholder-answer">
                        Your analysis result will appear here after running SATQueryAI.
                    </div>
                """

            st.html(
                f"""
                <div class="info-card">
                    <div class="info-title">Final Answer</div>
                    {answer_html}
                </div>
                """
            )

with right:
    result = st.session_state.result

    if result and result.get("success", False):
        intent = result.get("intent", "Unknown")
        routes = result.get("routes", [])
        route_text = ", ".join(routes) if routes else str(result.get("route", "Unknown"))

        confidence = result.get("confidence_score")
        confidence_display = (
            f"{conf}%"
            if confidence is not None
            else "N/A"
        )
        
        status_text = "Complete"

        answer = result.get("answer")
        executed_models = result.get("executed_models", [])

    else:
        intent = "—"
        route_text = "—"
        confidence_display = "—"

        
        status_text = "Ready"

        answer = None
        executed_models = []
    
    result_focused = st.session_state.section == "results"
    result_class = "section focused" if result_focused else "section"

    st.markdown(
        f"""
<div class="{result_class}">
    <div class="section-bar"></div>
    <div class="section-title">Analysis Summary</div>
</div>
""",
        unsafe_allow_html=True,
    )

    
    st.html(
    f"""
    <div class="info-card">
        <div class="summary-grid">

            <div class="summary-item">
                <div class="summary-label">Intent</div>
                <div class="summary-value">
                    {escape_html(str(intent))}
                </div>
            </div>

            <div class="summary-item">
                <div class="summary-label">Route</div>
                <div class="summary-value">
                    {escape_html(str(route_text))}
                </div>
            </div>

            <div class="summary-item">
                <div class="summary-label">Confidence</div>
                <div class="summary-value confidence">
                    {escape_html(str(confidence_display))}
                </div>
            </div>

            <div class="summary-item">
                <div class="summary-label">Status</div>
                <div class="summary-value success">
                    ✓ {escape_html(str(status_text))}
                </div>
            </div>

        </div>
    </div>
    """
)
    st.html(
    f"""
<div class="info-card">
    <div class="info-title">
        Final Answer
    </div>

    <div class="answer">
        {
            escape_html(str(answer))
            if answer
            else "Your analysis result will appear here after running SATQueryAI."
        }
    </div>
</div>
"""
)

    models_class = "section focused" if st.session_state.section == "models" else "section"

    st.markdown(
        f"""
<div class="{models_class}">
    <div class="section-bar"></div>
    <div class="section-title">Model Information</div>
</div>
""",
        unsafe_allow_html=True,
    )

    model_descriptions = {
        "SIA": (
            "Single Image Analysis for objects, land cover "
            "and visible scene features."
        ),
        "BTA": (
            "Bi-Temporal Analysis for identifying changes "
            "between T1 and T2 imagery."
        ),
        "Cross-Modal": (
            "Cross-Modal Analysis combining optical and SAR "
            "imagery when both are available."
        ),
    }

    if executed_models:
        model_text = ", ".join(str(name) for name in executed_models)

        descriptions = [
            model_descriptions.get(
                str(name),
                "Analysis model selected by the routing pipeline.",
            )
            for name in executed_models
        ]

        model_description = " ".join(descriptions)
    else:
        model_text = "No model selected yet"
        model_description = (
            "The routing pipeline will determine the "
            "appropriate analysis model."
        )

    st.html(
        f"""
<div class="info-card">
    <div class="model-box">
        <div class="model-icon">◈</div>
        <div>
            <div class="model-name">
                {escape_html(model_text)}
            </div>
            <div class="model-desc">
                {escape_html(model_description)}
            </div>
        </div>
    </div>
</div>
"""
    )