"""
app.py — Streamlit interface for DR Grading System.
Run with: streamlit run app.py
"""
import streamlit as st
from PIL import Image
import database as db
import image_checks
import inference

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="DR Grading System",
    page_icon="🏥",
    layout="wide",
)

# ====================== PREMIUM STYLING ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Overall background ── */
.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #fafbff 50%, #f0f7ff 100%);
    min-height: 100vh;
}

/* ── Header hero strip ── */
.hero-header {
    background: linear-gradient(135deg, #1a56db 0%, #0e9f6e 100%);
    border-radius: 18px;
    padding: 32px 40px 28px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(26,86,219,0.18);
}
.hero-header h1 {
    color: white;
    font-size: 2.1em;
    font-weight: 700;
    margin: 0 0 6px 0;
}
.hero-header p {
    color: rgba(255,255,255,0.85);
    font-size: 0.97em;
    margin: 0;
}

/* ── Safety banner ── */
.safety-banner {
    background: linear-gradient(90deg, #fff7e6, #fffbe6);
    border-left: 5px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 13px 20px;
    margin-bottom: 22px;
    color: #92400e;
    font-size: 0.93em;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(245,158,11,0.10);
}

/* ── Cards ── */
.card {
    background: white;
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: 0 2px 16px rgba(30,58,138,0.07);
    border: 1px solid #e8edf5;
    margin-bottom: 16px;
}

/* ── Grade badges ── */
.grade-badge {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.88em;
    letter-spacing: 0.02em;
    color: white;
}
.grade-0 { background: linear-gradient(90deg,#059669,#10b981); }
.grade-1 { background: linear-gradient(90deg,#16a34a,#4ade80); color:#14532d; }
.grade-2 { background: linear-gradient(90deg,#d97706,#fbbf24); color:#451a03; }
.grade-3 { background: linear-gradient(90deg,#dc2626,#f87171); }
.grade-4 { background: linear-gradient(90deg,#7c3aed,#c026d3); }

/* ── Metric tiles ── */
.metric-tile {
    background: linear-gradient(135deg,#eff6ff,#f0fdf4);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    border: 1px solid #dbeafe;
    box-shadow: 0 1px 6px rgba(59,130,246,0.07);
}
.metric-tile .value {
    font-size: 1.9em;
    font-weight: 700;
    color: #1e40af;
}
.metric-tile .label {
    font-size: 0.8em;
    color: #6b7280;
    margin-top: 2px;
}

/* ── Warn box ── */
.warn-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0;
    color: #92400e;
    font-size: 0.93em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 10px 10px 0 0;
    padding: 10px 24px;
    font-weight: 500;
    color: #374151;
    border: 1px solid #e5e7eb;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#1a56db,#0e9f6e) !important;
    color: white !important;
    border-color: transparent !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#1a56db,#0e9f6e);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1em;
    padding: 10px;
    box-shadow: 0 4px 14px rgba(26,86,219,0.25);
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(26,86,219,0.35);
}

/* ── Input fields — force light theme ── */
input, textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus {
    border-color: #1a56db !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.12) !important;
    outline: none !important;
}

/* stTextInput wrapper */
.stTextInput > div > div {
    background-color: #ffffff !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important;
}
.stTextInput > div > div:focus-within {
    border-color: #1a56db !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.12) !important;
}

/* stNumberInput wrapper */
.stNumberInput > div > div {
    background-color: #ffffff !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important;
}
.stNumberInput > div > div:focus-within {
    border-color: #1a56db !important;
}
.stNumberInput input {
    background-color: #ffffff !important;
    color: #111827 !important;
}

/* Selectbox closed state */
.stSelectbox > div > div {
    background-color: #ffffff !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #111827 !important;
}
.stSelectbox span {
    color: #111827 !important;
}
/* Selectbox dropdown popup (renders in a portal overlay) */
[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #dbeafe !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(26,86,219,0.12) !important;
}
li[role="option"] {
    background-color: #ffffff !important;
    color: #111827 !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #eff6ff !important;
    color: #1e40af !important;
}

/* File uploader */
.stFileUploader > div {
    background-color: #f8faff !important;
    border: 2px dashed #93c5fd !important;
    border-radius: 12px !important;
    color: #374151 !important;
}

/* Labels */
label, .stTextInput label, .stNumberInput label,
.stSelectbox label, .stFileUploader label {
    color: #374151 !important;
    font-weight: 500 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #f8faff !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 10px !important;
    color: #1e3a8a !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background-color: #ffffff !important;
    border: 1px solid #e0e7ff !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* Form container */
[data-testid="stForm"] {
    background: #f8faff !important;
    border: 1px solid #e0e7ff !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

/* General text color — but NOT inside the hero header */
p, span, div, h1, h2, h3, h4 {
    color: #111827;
}
.hero-header h1, .hero-header p, .hero-header * {
    color: white !important;
}

/* ── Secondary / default buttons (Refresh, Delete, etc.) ── */
.stButton > button {
    background-color: #ffffff !important;
    color: #1a56db !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: background 0.15s, border-color 0.15s !important;
}
.stButton > button:hover {
    background-color: #eff6ff !important;
    border-color: #1a56db !important;
    color: #1e40af !important;
}
/* Keep the primary Analyse button gradient override */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#1a56db,#0e9f6e) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(26,86,219,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(26,86,219,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

GRADE_COLORS = {0: "grade-0", 1: "grade-1", 2: "grade-2", 3: "grade-3", 4: "grade-4"}
CONFIDENCE_THRESHOLD = 0.55


def grade_badge(grade: int, grade_str: str) -> str:
    cls = GRADE_COLORS.get(grade, "grade-4")
    return f'<span class="grade-badge {cls}">{grade_str}</span>'


# ====================== HERO HEADER ======================
st.markdown("""
<div class="hero-header">
    <h1>Diabetic Retinopathy Grading System</h1>
    <p>AI-Powered Retinal Screening · IDRiD Dataset · EfficientNet-B0 · QWK 0.8257</p>
</div>
""", unsafe_allow_html=True)

# ── Safety disclaimer ──────────────────────────────────────────────────────────
st.markdown("""
<div class="safety-banner">
    ⚕️ <strong>Screening Support – Non-Diagnostic.</strong>
    This tool assists trained healthcare workers in triage screening only.
    It does <strong>not</strong> replace clinical examination by a qualified ophthalmologist.
    All results must be reviewed by a licensed medical professional before any clinical action.
</div>
""", unsafe_allow_html=True)

tab_upload, tab_patients = st.tabs(["  New Patient Analysis  ", "  Patient Registry  "])


# ─────────────────────────── TAB 1: UPLOAD ────────────────────────────────────
with tab_upload:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col_form, col_result = st.columns([1, 1.5], gap="large")

    with col_form:
        st.markdown("### Patient Details")
        with st.form("patient_form", clear_on_submit=False):
            patient_id = st.text_input("Patient ID", placeholder="e.g. PT-2024-001")
            name       = st.text_input("Full Name",  placeholder="e.g. Ramesh Kumar")
            age        = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)

            st.markdown("---")
            st.markdown("### Fundal Image")
            uploaded = st.file_uploader("Upload fundal photograph (JPG / PNG)",
                                        type=["jpg", "jpeg", "png"])

            submitted = st.form_submit_button("Run Analysis", use_container_width=True,
                                              type="primary")

    with col_result:
        if submitted:
            if not patient_id.strip() or not name.strip():
                st.error("Please fill in Patient ID and Full Name.")
            elif uploaded is None:
                st.error("Please upload a fundal image.")
            else:
                pil_img = Image.open(uploaded)

                with st.spinner("Checking image quality…"):
                    ok_quality, quality_msg = image_checks.check_quality(pil_img)
                if not ok_quality:
                    st.error(quality_msg)
                    st.stop()

                with st.spinner("Verifying fundal image…"):
                    ok_fundal, fundal_msg = image_checks.is_fundal(pil_img)
                if not ok_fundal:
                    st.error(fundal_msg)
                    st.stop()

                with st.spinner("Running DR grading model…"):
                    result = inference.predict(pil_img)

                grade      = result["grade"]
                grade_str  = result["grade_str"]
                confidence = result["confidence"]
                result_img = result["result_image"]

                db.add_patient(
                    name=name.strip(), age=int(age), patient_id=patient_id.strip(),
                    grade=grade, grade_str=grade_str, confidence=confidence,
                    original_image=pil_img, result_image=result_img,
                )

                st.markdown("### Analysis Result")
                st.image(result_img, caption="Lesion regions highlighted (yellow = high attention, green = boundary)",
                         use_container_width=True)

                # Metric tiles
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"""
                    <div class="metric-tile">
                        <div class="value">{grade_badge(grade, grade_str)}</div>
                        <div class="label">DR Grade</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    conf_color = "#059669" if confidence >= CONFIDENCE_THRESHOLD else "#dc2626"
                    st.markdown(f"""
                    <div class="metric-tile">
                        <div class="value" style="color:{conf_color}">{confidence:.1%}</div>
                        <div class="label">Model Confidence</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if confidence < CONFIDENCE_THRESHOLD:
                    st.markdown('<div class="warn-box">⚠️ <strong>Low confidence prediction.</strong> '
                                'This result should be confirmed by a qualified ophthalmologist '
                                'before any clinical decisions are made.</div>', unsafe_allow_html=True)

                if grade == 0:
                    st.success("No signs of diabetic retinopathy detected. Routine annual screening recommended.")
                elif grade == 1:
                    st.info("Mild DR detected. Follow-up in 12 months recommended.")
                elif grade == 2:
                    st.warning("Moderate DR detected. Refer to ophthalmologist within 3–6 months.")
                elif grade == 3:
                    st.error("Severe DR detected. Urgent referral to ophthalmologist recommended.")
                elif grade == 4:
                    st.error("Proliferative DR detected. Immediate ophthalmologist referral required.")

                st.success(f"Patient record saved for **{name.strip()}**.")

                # ── Display Similar Cases ──
                similar_cases = result.get("similar_cases", [])
                if similar_cases:
                    st.markdown("---")
                    st.markdown("### Visually Similar Cases")
                    st.markdown("<p style='color:#6b7280; font-size:0.9em; margin-top:-10px;'>These verified cases from the training database share similar retinal features.</p>", unsafe_allow_html=True)
                    sc_cols = st.columns(len(similar_cases))
                    for i, sc in enumerate(similar_cases):
                        with sc_cols[i]:
                            st.image(sc["image"], caption=f"Match {i+1} ({sc['filename']})", use_container_width=True)

        else:
            st.markdown("### Analysis Result")
            st.info("Complete the form and upload a fundal image, then click **Run Analysis**.")

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────── TAB 2: PATIENT LIST ──────────────────────────────
with tab_patients:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Patient Registry")

    col_sort, col_refresh = st.columns([3, 1])
    with col_sort:
        sort_option = st.selectbox("Sort by", options=["severity_desc", "severity_asc", "date_desc"],
            format_func=lambda x: {
                "severity_desc": "Severity — High to Low",
                "severity_asc":  "Severity — Low to High",
                "date_desc":     "Date — Newest First",
            }[x])
    with col_refresh:
        st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
        if st.button("Refresh", use_container_width=True):
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    patients = db.get_all_patients(sort_by=sort_option)

    if not patients:
        st.info("No patient records yet. Analyse a patient in the **New Patient Analysis** tab.")
    else:
        # Summary stats
        s1, s2, s3, s4 = st.columns(4)
        urgent = sum(1 for p in patients if p["grade"] >= 3)
        s1.metric("Total Patients", len(patients))
        s2.metric("Urgent Referrals", urgent)
        s3.metric("Avg Confidence", f"{sum(p['confidence'] for p in patients)/len(patients):.1%}")
        s4.metric("Low Confidence", sum(1 for p in patients if p["confidence"] < CONFIDENCE_THRESHOLD))

        st.markdown("---")

        for p in patients:
            conf_pct = f"{p['confidence']:.1%}"
            low_conf = p["confidence"] < CONFIDENCE_THRESHOLD

            label = f"{p['name']}  ·  {p['patient_id']}  ·  {p['grade_str']}  ·  {conf_pct}"
            if low_conf:
                label += "  ⚠️ Low Confidence"

            with st.expander(label, expanded=False):
                img_col, info_col = st.columns([1, 1], gap="large")

                with img_col:
                    try:
                        result_img = Image.open(p["result_path"])
                        st.image(result_img, caption="Lesion highlight", use_container_width=True)
                    except Exception:
                        st.warning("Result image not found.")

                with info_col:
                    st.markdown(
                        f"**Name:** {p['name']}  \n"
                        f"**Patient ID:** {p['patient_id']}  \n"
                        f"**Age:** {p['age']}  \n"
                        f"**Date:** {p['date']}  \n\n"
                        f"**Diagnosis:** {grade_badge(p['grade'], p['grade_str'])}  \n\n"
                        f"**Confidence:** `{conf_pct}`",
                        unsafe_allow_html=True,
                    )

                    if low_conf:
                        st.markdown('<div class="warn-box">⚠️ <strong>Low confidence.</strong> '
                                    'Clinical review by an ophthalmologist is recommended.</div>',
                                    unsafe_allow_html=True)

                    if st.button("Delete record", key=f"del_{p['record_id']}"):
                        db.delete_patient(p["record_id"])
                        st.success("Record deleted.")
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
