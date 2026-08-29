"""
SATQueryAI - Streamlit Frontend
================================

User interface for SATQueryAI.

Flow:

    User
      |
      +---- Query
      |
      +---- T1 Image
      |
      +---- T2 Image
      |
      +---- SAR Image
      |
      v
    app.py
      |
      v
    Routing Graph
      |
      v
    SIA / BTA / Cross-Modal
      |
      v
    Evidence + Confidence
      |
      v
    Result
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# BACKEND IMPORT
# ============================================================

from app import process_query


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="SATQueryAI",

    page_icon="🛰️",

    layout="wide",

    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------
       Main background
    ------------------------------------------------------ */

    .stApp {
        background-color: #f5f9ff;
    }


    /* ------------------------------------------------------
       Main title
    ------------------------------------------------------ */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #075985;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #475569;
        margin-top: 0px;
        margin-bottom: 25px;
    }


    /* ------------------------------------------------------
       Section headers
    ------------------------------------------------------ */

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #0369a1;
        margin-top: 20px;
        margin-bottom: 10px;
    }


    /* ------------------------------------------------------
       Result cards
    ------------------------------------------------------ */

    .result-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #bae6fd;
        margin-bottom: 15px;
    }


    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #bae6fd;
        text-align: center;
    }


    .metric-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }


    .metric-value {
        color: #075985;
        font-size: 25px;
        font-weight: 800;
    }


    /* ------------------------------------------------------
       Answer box
    ------------------------------------------------------ */

    .answer-box {
        background: #ffffff;
        border-left: 5px solid #0284c7;
        padding: 20px;
        border-radius: 8px;
        color: #1e293b;
        font-size: 17px;
        line-height: 1.6;
    }


    /* ------------------------------------------------------
       Footer
    ------------------------------------------------------ */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:

    st.session_state.result = None


if "analysis_started" not in st.session_state:

    st.session_state.analysis_started = False


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">SATQueryAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Satellite Intelligence Query and Analysis System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## SATQueryAI"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Analysis Modes"
    )

    st.markdown(
        """
        **Single Image Analysis**

        Analyze objects, land cover, vegetation,
        roads, buildings and other visible features.

        **Bi-Temporal Analysis**

        Compare two satellite images and identify
        changes between T1 and T2.

        **Cross-Modal Analysis**

        Combine optical and Synthetic Aperture Radar
        imagery.

        **Complex Analysis**

        Combine multiple analysis sources.
        """
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Pipeline"
    )

    st.markdown(
        """
        Query
        ↓

        Intent Classification
        ↓

        Intelligent Routing
        ↓

        Satellite Models
        ↓

        Geospatial Analysis
        ↓

        Evidence Generation
        ↓

        Confidence Estimation
        """
    )

    st.markdown(
        "---"
    )

    st.caption(
        "SATQueryAI"
    )

    st.caption(
        "AI-powered satellite image analysis"
    )


# ============================================================
# QUERY SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Ask SATQueryAI</div>',
    unsafe_allow_html=True
)

query = st.text_area(

    "Enter your query",

    placeholder=(
        "Example: What changed between T1 and T2?"
    ),

    height=100
)


# ============================================================
# IMAGE UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Upload Satellite Data</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


# ============================================================
# T1
# ============================================================

with col1:

    t1_file = st.file_uploader(

        "T1 / Image",

        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ],

        key="t1_upload",

        help=(
            "Upload the first satellite image."
        )
    )


# ============================================================
# T2
# ============================================================

with col2:

    t2_file = st.file_uploader(

        "T2 / Second Image",

        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff"
        ],

        key="t2_upload",

        help=(
            "Upload the second image for "
            "bi-temporal analysis."
        )
    )


# ============================================================
# SAR
# ============================================================

sar_file = st.file_uploader(

    "SAR Image (optional)",

    type=[
        "tif",
        "tiff",
        "png",
        "jpg",
        "jpeg"
    ],

    key="sar_upload",

    help=(
        "Upload a SAR image when the query "
        "requires optical + SAR analysis."
    )
)


# ============================================================
# IMAGE PREVIEW
# ============================================================

if t1_file or t2_file or sar_file:

    st.markdown(
        '<div class="section-title">Image Preview</div>',
        unsafe_allow_html=True
    )


preview_columns = st.columns(3)


# ------------------------------------------------------------
# T1 Preview
# ------------------------------------------------------------

with preview_columns[0]:

    if t1_file:

        st.markdown(
            "**T1**"
        )

        try:

            st.image(
                t1_file,
                use_container_width=True
            )

        except Exception:

            st.warning(
                "Preview unavailable for this file."
            )


# ------------------------------------------------------------
# T2 Preview
# ------------------------------------------------------------

with preview_columns[1]:

    if t2_file:

        st.markdown(
            "**T2**"
        )

        try:

            st.image(
                t2_file,
                use_container_width=True
            )

        except Exception:

            st.warning(
                "Preview unavailable for this file."
            )


# ------------------------------------------------------------
# SAR Preview
# ------------------------------------------------------------

with preview_columns[2]:

    if sar_file:

        st.markdown(
            "**SAR**"
        )

        try:

            st.image(
                sar_file,
                use_container_width=True
            )

        except Exception:

            st.warning(
                "Preview unavailable for this file."
            )


# ============================================================
# FILE SAVING
# ============================================================

def save_uploaded_file(
    uploaded_file,
    directory,
    filename
):
    """
    Save a Streamlit UploadedFile to disk.
    """

    directory = Path(
        directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        directory
        /
        filename
    )

    with open(
        file_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    return str(
        file_path
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown(
    ""
)

analyze_button = st.button(

    "Analyze Satellite Data",

    type="primary",

    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query.strip():

        st.error(
            "Please enter a query."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not t1_file and not t2_file and not sar_file:

        st.error(
            "Please upload at least one satellite image."
        )

        st.stop()


    # --------------------------------------------------------
    # Temporary input directory
    # --------------------------------------------------------

    input_directory = (
        PROJECT_ROOT
        /
        "data"
        /
        "input"
    )

    input_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    t1_path = None
    t2_path = None
    sar_path = None


    try:

        if t1_file:

            t1_path = save_uploaded_file(

                t1_file,

                input_directory,

                "uploaded_T1"
                +
                Path(
                    t1_file.name
                ).suffix
            )


        if t2_file:

            t2_path = save_uploaded_file(

                t2_file,

                input_directory,

                "uploaded_T2"
                +
                Path(
                    t2_file.name
                ).suffix
            )


        if sar_file:

            sar_path = save_uploaded_file(

                sar_file,

                input_directory,

                "uploaded_SAR"
                +
                Path(
                    sar_file.name
                ).suffix
            )


    except Exception as exc:

        st.error(
            f"Could not save uploaded files: {exc}"
        )

        st.stop()


    # --------------------------------------------------------
    # Determine optical input
    # --------------------------------------------------------

    optical_path = t1_path


    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    st.session_state.analysis_started = True

    st.session_state.result = None


    with st.status(
        "Running SATQueryAI...",
        expanded=True
    ) as status:

        st.write(
            "Classifying query..."
        )

        st.write(
            "Selecting analysis route..."
        )

        st.write(
            "Running satellite analysis..."
        )

        st.write(
            "Generating evidence..."
        )

        st.write(
            "Calculating confidence..."
        )


        try:

            result = process_query(

                query=query,

                image_path=t1_path,

                optical_path=optical_path,

                sar_path=sar_path,

                t1_path=t1_path,

                t2_path=t2_path,
            )

            st.session_state.result = result

            status.update(
                label="Analysis completed",
                state="complete",
                expanded=False
            )


        except Exception as exc:

            status.update(
                label="Analysis failed",
                state="error",
                expanded=True
            )

            st.error(
                str(exc)
            )

            st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

result = st.session_state.result


if result:

    st.markdown(
        "---"
    )

    st.markdown(
        '<div class="section-title">Analysis Result</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # ERROR
    # ========================================================

    if not result.get(
        "success",
        False
    ):

        st.error(
            result.get(
                "error",
                "Unknown error."
            )
        )

        if result.get(
            "errors"
        ):

            with st.expander(
                "Technical Details"
            ):

                for error in result[
                    "errors"
                ]:

                    st.write(
                        error
                    )

        st.stop()


    # ========================================================
    # TOP METRICS
    # ========================================================

    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        st.metric(

            "Intent",

            result.get(
                "intent",
                "Unknown"
            )
        )


    with metric2:

        routes = result.get(
            "routes",
            []
        )

        route_text = (
            ", ".join(routes)
            if routes
            else
            str(
                result.get(
                    "route",
                    "Unknown"
                )
            )
        )

        st.metric(

            "Route",

            route_text
        )


    with metric3:

        confidence = result.get(
            "confidence_score"
        )

        if confidence is not None:

            confidence_display = (
                f"{float(confidence):.3f}"
            )

        else:

            confidence_display = "N/A"

        st.metric(

            "Confidence",

            confidence_display
        )


    with metric4:

        st.metric(

            "Status",

            result.get(
                "execution_status",
                "Unknown"
            )
        )


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    st.markdown(
        '<div class="section-title">Final Answer</div>',
        unsafe_allow_html=True
    )


    answer = result.get(
        "answer"
    )


    if answer:

        st.markdown(
            f"""
            <div class="answer-box">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info(
            "No final answer was generated."
        )


    # ========================================================
    # ROUTING INFORMATION
    # ========================================================

    with st.expander(
        "Routing Information",
        expanded=True
    ):

        routing_col1, routing_col2 = st.columns(2)


        with routing_col1:

            st.write(
                "**Intent:**",
                result.get(
                    "intent"
                )
            )

            st.write(
                "**Intent Confidence:**",
                result.get(
                    "intent_confidence"
                )
            )

            st.write(
                "**Intent Reason:**",
                result.get(
                    "intent_reason"
                )
            )


        with routing_col2:

            st.write(
                "**Route:**",
                result.get(
                    "route"
                )
            )

            st.write(
                "**Routes:**",
                result.get(
                    "routes"
                )
            )

            st.write(
                "**Routing Confidence:**",
                result.get(
                    "routing_confidence"
                )
            )

            st.write(
                "**Routing Reason:**",
                result.get(
                    "routing_reason"
                )
            )


    # ========================================================
    # MODEL RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">Model Analysis</div>',
        unsafe_allow_html=True
    )


    executed_models = result.get(
        "executed_models",
        []
    )


    if executed_models:

        for model_name in executed_models:

            if model_name == "SIA":

                model_answer = result.get(
                    "sia_answer"
                )

            elif model_name == "BTA":

                model_answer = result.get(
                    "bta_answer"
                )

            elif model_name == "Cross-Modal":

                model_answer = result.get(
                    "cross_modal_answer"
                )

            else:

                model_answer = None


            with st.expander(
                f"{model_name} Analysis",
                expanded=True
            ):

                if model_answer:

                    st.write(
                        model_answer
                    )

                else:

                    st.info(
                        "No output available."
                    )

    else:

        st.info(
            "No model execution information available."
        )


    # ========================================================
    # EVIDENCE
    # ========================================================

    st.markdown(
        '<div class="section-title">Evidence</div>',
        unsafe_allow_html=True
    )


    evidence = result.get(
        "evidence"
    )


    evidence_items = result.get(
        "evidence_items",
        []
    )


    if isinstance(
        evidence,
        dict
    ):

        evidence_items = evidence.get(
            "evidence",
            evidence_items
        )


    if evidence_items:

        for index, item in enumerate(
            evidence_items,
            start=1
        ):

            if isinstance(
                item,
                dict
            ):

                source = item.get(
                    "source",
                    "Unknown"
                )

                observation = item.get(
                    "observation",
                    item.get(
                        "text",
                        ""
                    )
                )

                strength = item.get(
                    "strength",
                    "unknown"
                )

                st.markdown(
                    f"""
                    **{index}. {source}**

                    {observation}

                    Strength: **{strength}**
                    """
                )

            else:

                st.write(
                    f"{index}. {item}"
                )

    else:

        st.info(
            "No structured evidence available."
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    st.markdown(
        '<div class="section-title">Confidence Assessment</div>',
        unsafe_allow_html=True
    )


    confidence_col1, confidence_col2 = st.columns(2)


    with confidence_col1:

        confidence_score = result.get(
            "confidence_score"
        )

        if confidence_score is not None:

            st.metric(

                "Confidence Score",

                f"{float(confidence_score):.3f}"
            )

        else:

            st.metric(
                "Confidence Score",
                "N/A"
            )


    with confidence_col2:

        st.metric(

            "Confidence Level",

            result.get(
                "confidence_level",
                "N/A"
            )
        )


    # --------------------------------------------------------
    # Confidence details
    # --------------------------------------------------------

    confidence_data = result.get(
        "confidence"
    )


    if isinstance(
        confidence_data,
        dict
    ):

        with st.expander(
            "Confidence Components"
        ):

            for key, value in confidence_data.items():

                if key in [
                    "confidence_score",
                    "confidence_level"
                ]:

                    continue

                st.write(
                    f"**{key}:** {value}"
                )


    # ========================================================
    # EXECUTION INFORMATION
    # ========================================================

    with st.expander(
        "Execution Information"
    ):

        st.write(
            "**Execution Status:**",
            result.get(
                "execution_status"
            )
        )

        st.write(
            "**Execution Time:**",
            result.get(
                "execution_time"
            ),
            "seconds"
        )

        st.write(
            "**Executed Models:**",
            result.get(
                "executed_models"
            )
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        SATQueryAI · Satellite Intelligence Query and Analysis System
    </div>
    """,
    unsafe_allow_html=True
)