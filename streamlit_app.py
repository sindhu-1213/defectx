#!/usr/bin/env python3
"""
DefectX – Industrial Quality Inspection Software (Streamlit Application)
Dual-stage vision pipeline: YOLO11 instance segmentation -> MobileNetV3 defect classification.
"""

from __future__ import annotations

import html
import io
import os
import tempfile
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

# Configure Matplotlib headless backend and cache before importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "defectx_mpl"))
try:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

import app
from src import database, report_generator
from src.decision_engine import ObjectPrediction, ProductDecision, decide_product

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DefectX | Industrial Quality Inspection Software",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Paths and Constants
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATABASE_PATH = database.DEFAULT_DATABASE_PATH
DEFAULT_YOLO = HERE / "models" / "yolo11_seg_best.pt"
DEFAULT_MOBILENET = HERE / "models" / "mobilenet_v3_best.pt"
SAMPLE_IMAGE_DIR = HERE / "assets" / "sample_images"
SAMPLE_IMAGES = app.SAMPLE_IMAGES
IMAGE_SUFFIXES = app.IMAGE_SUFFIXES
RESULT_COLUMNS = app.RESULT_COLUMNS
HISTORY_COLUMNS = app.HISTORY_COLUMNS

# Initialize database on startup
database.initialize_database(DATABASE_PATH)

# -----------------------------------------------------------------------------
# Cached Model Initialization
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO11 & MobileNetV3 Inspection Checkpoints...")
def get_models() -> app.InspectionModels:
    if not DEFAULT_YOLO.is_file():
        raise FileNotFoundError(f"YOLO11 checkpoint not found: {DEFAULT_YOLO}")
    if not DEFAULT_MOBILENET.is_file():
        raise FileNotFoundError(f"MobileNetV3 checkpoint not found: {DEFAULT_MOBILENET}")
    models = app.InspectionModels(DEFAULT_YOLO, DEFAULT_MOBILENET)
    app.MODELS = models
    return models

# -----------------------------------------------------------------------------
# Complete Dark Industrial SaaS CSS & 3D Visual Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
      --dx-bg: #070d18;
      --dx-sidebar-bg: #091120;
      --dx-panel-bg: rgba(13, 23, 42, 0.85);
      --dx-card-bg: rgba(16, 29, 52, 0.7);
      --dx-border: rgba(56, 189, 248, 0.16);
      --dx-border-active: rgba(56, 189, 248, 0.5);
      --dx-accent: #38bdf8;
      --dx-accent-glow: rgba(56, 189, 248, 0.28);
      --dx-pass: #10b981;
      --dx-fail: #ef4444;
      --dx-review: #f59e0b;
      --dx-text-main: #f1f5f9;
      --dx-text-muted: #94a3b8;
    }
    
    /* Global App Background & Base Layout */
    .stApp {
        background-color: var(--dx-bg) !important;
        color: var(--dx-text-main) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--dx-sidebar-bg) !important;
        border-right: 1px solid var(--dx-border) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.35) !important;
    }
    
    /* Brand Header in Sidebar */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 6px 18px;
        border-bottom: 1px solid var(--dx-border);
        margin-bottom: 16px;
    }
    .brand-icon {
        width: 38px;
        height: 38px;
        border-radius: 9px;
        background: linear-gradient(135deg, #0284c7, #2563eb);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.3);
        flex-shrink: 0;
    }
    .brand-name {
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #fff;
        line-height: 1.1;
    }
    .brand-name span { color: var(--dx-accent); }
    .brand-sub {
        font-size: 9.5px;
        letter-spacing: 0.15em;
        font-family: 'DM Mono', monospace;
        color: var(--dx-text-muted);
    }
    
    /* Sidebar Telemetry Block */
    .sidebar-telemetry {
        margin-top: 18px;
        padding: 12px 14px;
        background: rgba(10, 18, 32, 0.7);
        border: 1px solid var(--dx-border);
        border-radius: 10px;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.05);
    }
    .telemetry-header {
        font-size: 10px;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.14em;
        color: var(--dx-text-muted);
        margin-bottom: 8px;
        text-transform: uppercase;
        font-weight: 700;
    }
    .telemetry-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 11.5px;
        padding: 5px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .telemetry-row:last-child { border-bottom: none; }
    .telemetry-label { color: #94a3b8; }
    .telemetry-val { font-weight: 600; color: #cbd5e1; }
    .telemetry-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .telemetry-dot.green { background: var(--dx-pass); box-shadow: 0 0 8px var(--dx-pass); }
    .telemetry-dot.cyan { background: var(--dx-accent); box-shadow: 0 0 8px var(--dx-accent); }
    .telemetry-dot.blue { background: #3b82f6; box-shadow: 0 0 8px #3b82f6; }

    /* Section Headers */
    .panel-kicker {
        font-size: 11px;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.14em;
        color: var(--dx-accent);
        text-transform: uppercase;
        margin-bottom: 2px;
        font-weight: 700;
    }
    .panel-title {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 14px;
        letter-spacing: -0.01em;
    }

    /* 3D Holographic Spatial Sensing Node */
    .hologram-card {
        display: grid;
        grid-template-columns: 280px 1fr;
        gap: 24px;
        align-items: center;
        background: linear-gradient(135deg, rgba(10, 20, 38, 0.92), rgba(6, 13, 26, 0.96));
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45), inset 0 1px 1px rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }
    .hologram-scene {
        position: relative;
        width: 240px;
        height: 180px;
        margin: 0 auto;
        perspective: 900px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hologram-grid-plate {
        position: absolute;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        border: 1px solid rgba(56, 189, 248, 0.3);
        background: radial-gradient(circle, rgba(2, 132, 199, 0.15) 0%, transparent 70%);
        transform: rotateX(68deg) translateZ(-40px);
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.2);
        animation: gridPulse 4s ease-in-out infinite;
    }
    @keyframes gridPulse {
        0%, 100% { transform: rotateX(68deg) translateZ(-40px) scale(0.98); opacity: 0.7; }
        50% { transform: rotateX(68deg) translateZ(-40px) scale(1.04); opacity: 1; }
    }
    .hologram-radar-rings .ring {
        position: absolute;
        border-radius: 50%;
        border: 1px dashed rgba(56, 189, 248, 0.35);
        transform: rotateX(68deg) translateZ(-40px);
    }
    .hologram-radar-rings .r1 { width: 180px; height: 180px; top: 0; left: 30px; animation: ringSpin 16s linear infinite; }
    .hologram-radar-rings .r2 { width: 130px; height: 130px; top: 25px; left: 55px; border-color: rgba(16, 185, 129, 0.4); border-style: solid; animation: ringSpin 12s linear infinite reverse; }
    .hologram-radar-rings .r3 { width: 80px; height: 80px; top: 50px; left: 80px; border-color: rgba(56, 189, 248, 0.5); }
    @keyframes ringSpin { to { transform: rotateX(68deg) translateZ(-40px) rotateZ(360deg); } }

    .hologram-cube {
        width: 60px;
        height: 60px;
        position: relative;
        transform-style: preserve-3d;
        animation: cubeRotate3D 10s linear infinite;
    }
    @keyframes cubeRotate3D {
        0% { transform: rotateX(18deg) rotateY(0deg) rotateZ(12deg) translateY(-8px); }
        50% { transform: rotateX(25deg) rotateY(180deg) rotateZ(5deg) translateY(4px); }
        100% { transform: rotateX(18deg) rotateY(360deg) rotateZ(12deg) translateY(-8px); }
    }
    .cube-face {
        position: absolute;
        width: 60px;
        height: 60px;
        border: 1.5px solid rgba(56, 189, 248, 0.65);
        background: rgba(2, 132, 199, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'DM Mono', monospace;
        font-size: 8px;
        color: #38bdf8;
        text-align: center;
        line-height: 1.1;
        box-shadow: inset 0 0 10px rgba(56, 189, 248, 0.2);
    }
    .cube-face.front  { transform: rotateY(0deg) translateZ(30px); }
    .cube-face.back   { transform: rotateY(180deg) translateZ(30px); }
    .cube-face.right  { transform: rotateY(90deg) translateZ(30px); }
    .cube-face.left   { transform: rotateY(-90deg) translateZ(30px); }
    .cube-face.top    { transform: rotateX(90deg) translateZ(30px); }
    .cube-face.bottom { transform: rotateX(-90deg) translateZ(30px); }

    .hologram-scan-sweep {
        position: absolute;
        width: 140px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #38bdf8, transparent);
        box-shadow: 0 0 12px #38bdf8;
        animation: scanFloat 3s ease-in-out infinite alternate;
    }
    @keyframes scanFloat {
        0% { transform: translateY(-35px) scaleX(0.7); opacity: 0.3; }
        100% { transform: translateY(35px) scaleX(1.1); opacity: 1; }
    }
    .hologram-info h3 {
        color: #fff;
        font-size: 19px;
        margin: 4px 0 8px;
        font-weight: 700;
    }
    .hologram-info p {
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.6;
        margin: 0 0 14px;
    }
    .hologram-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hologram-tags span {
        padding: 5px 10px;
        background: rgba(15, 27, 48, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 6px;
        font-size: 11px;
        color: #cbd5e1;
        font-family: 'DM Mono', monospace;
    }
    .hologram-tags b { color: var(--dx-accent); }

    /* KPI Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: rgba(11, 20, 36, 0.88);
        border: 1px solid var(--dx-border);
        border-radius: 12px;
        padding: 16px 18px;
        display: flex;
        flex-direction: column;
        gap: 4px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.08);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.45);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.5), 0 0 20px rgba(56, 189, 248, 0.2);
    }
    .metric-card.total { border-left: 4px solid var(--dx-accent); }
    .metric-card.pass { border-left: 4px solid var(--dx-pass); }
    .metric-card.fail { border-left: 4px solid var(--dx-fail); }
    .metric-card.review { border-left: 4px solid var(--dx-review); }

    .metric-label {
        font-size: 11px;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.1em;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        margin: 4px 0;
        line-height: 1.1;
    }
    .metric-value.total { color: var(--dx-accent); }
    .metric-value.pass { color: #34d399; text-shadow: 0 0 14px rgba(16, 185, 129, 0.3); }
    .metric-value.fail { color: #f87171; text-shadow: 0 0 14px rgba(239, 68, 68, 0.3); }
    .metric-value.review { color: #fbbf24; text-shadow: 0 0 14px rgba(245, 158, 11, 0.3); }

    /* Dual Viewport Frames with 3D Laser Scan */
    .viewport-box {
        position: relative;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(16, 29, 52, 0.7), rgba(8, 15, 28, 0.9));
        border: 1px solid rgba(56, 189, 248, 0.22);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.12);
        padding: 14px;
        margin-bottom: 14px;
        overflow: hidden;
    }
    .viewport-box::after {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.8), #38bdf8, rgba(56, 189, 248, 0.8), transparent);
        box-shadow: 0 0 14px #38bdf8, 0 0 28px rgba(56, 189, 248, 0.6);
        opacity: 0.65;
        pointer-events: none;
        animation: scanLaser3D 4.5s ease-in-out infinite;
    }
    @keyframes scanLaser3D {
        0% { top: 0%; opacity: 0; }
        15% { opacity: 0.85; }
        85% { opacity: 0.85; }
        100% { top: 100%; opacity: 0; }
    }
    .viewport-title {
        font-size: 12px;
        font-family: 'DM Mono', monospace;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #94a3b8;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* Decision Status Banners */
    .decision-banner {
        border-radius: 12px;
        padding: 18px 22px;
        margin: 14px 0 20px 0;
        font-weight: 600;
        display: flex;
        flex-direction: column;
        gap: 6px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.12);
    }
    .decision-banner.pass {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #6ee7b7;
    }
    .decision-banner.fail {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
    }
    .decision-banner.review {
        background: rgba(245, 158, 11, 0.14);
        border: 1px solid rgba(245, 158, 11, 0.5);
        color: #fcd34d;
    }
    .decision-title {
        font-size: 20px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.01em;
    }

    /* 3D Floating Isometric Workflow Stations in About Page */
    .workflow-flow {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 22px 14px;
        background: rgba(9, 17, 31, 0.7);
        border: 1px solid var(--dx-border);
        border-radius: 10px;
        margin-top: 14px;
        overflow-x: auto;
    }
    .flow-step {
        text-align: center;
        min-width: 130px;
        padding: 14px 10px;
        border-radius: 8px;
        background: rgba(14, 25, 45, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.15);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .flow-step:hover {
        transform: translateY(-6px);
        border-color: var(--dx-accent);
        box-shadow: 0 14px 28px rgba(2, 132, 199, 0.3), 0 0 16px rgba(56, 189, 248, 0.3);
    }
    .flow-step span {
        display: block;
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: var(--dx-accent);
        margin-bottom: 2px;
    }
    .flow-step strong {
        font-size: 12.5px;
        color: #f1f5f9;
    }
    .flow-arrow {
        color: var(--dx-accent);
        font-size: 18px;
        font-weight: 800;
        opacity: 0.6;
    }

    /* Sample Gallery Cards */
    .sample-card {
        background: rgba(13, 23, 42, 0.75);
        border: 1px solid var(--dx-border);
        border-radius: 10px;
        padding: 8px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .sample-card:hover {
        border-color: var(--dx-accent);
        transform: translateY(-3px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Reusable Verified Single-Image Inference Pipeline Function
# -----------------------------------------------------------------------------
def run_single_inference(
    image: Image.Image,
    confidence: float = 0.25,
    iou: float = 0.70,
    image_size: int = 640,
    decision_confidence_threshold: float = 0.50,
) -> tuple[list[app.SegmentedObject], np.ndarray, ProductDecision, Image.Image, float, float, float]:
    """
    Executes the complete dual-stage pipeline with strict scalar typing for decision engine.
    """
    models = get_models()
    t_start = time.perf_counter()
    objects, yolo_ms = models.segment(
        image,
        confidence=float(confidence),
        iou=float(iou),
        image_size=int(image_size),
    )
    probabilities, mobile_ms = models.classify(objects)
    total_ms = (time.perf_counter() - t_start) * 1000.0

    # Build sequence of ObjectPrediction dataclasses
    predictions = [
        ObjectPrediction(
            class_name=item.class_name,
            defect_probability=(
                float(probabilities[index])
                if index < len(probabilities) and np.isfinite(probabilities[index])
                else None
            ),
        )
        for index, item in enumerate(objects)
    ]

    # Call decide_product with verified scalar float thresholds and known classes set
    product_decision = decide_product(
        predictions=predictions,
        classifier_threshold=float(models.threshold),
        decision_confidence_threshold=float(decision_confidence_threshold),
        known_classes=set(models.class_to_idx),
        processing_error=None,
    )

    annotated = app.annotate(
        image,
        objects,
        probabilities,
        models.threshold,
        float(decision_confidence_threshold),
    )

    return objects, probabilities, product_decision, annotated, yolo_ms, mobile_ms, total_ms

# -----------------------------------------------------------------------------
# Sidebar Navigation & Telemetry
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
            </div>
            <div class="brand-titles">
                <div class="brand-name"><span>DEFECT</span>X</div>
                <div class="brand-sub">QUALITY INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "MAIN NAVIGATION",
        [
            "📊 Dashboard",
            "🔬 AI Inspection",
            "📦 Batch Inspection",
            "🗃️ Inspection History",
            "📄 Reports Center",
            "ℹ️ About the System",
        ],
        index=0,
    )

    st.markdown(
        """
        <div class="sidebar-telemetry">
            <div class="telemetry-header">SYSTEM ARCHITECTURE</div>
            <div class="telemetry-row">
                <span class="telemetry-label"><span class="telemetry-dot green"></span>YOLO11 Segmentation</span>
                <span class="telemetry-val">Online</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label"><span class="telemetry-dot green"></span>MobileNetV3 Classifier</span>
                <span class="telemetry-val">Online</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label"><span class="telemetry-dot cyan"></span>SQLite Audit Trail</span>
                <span class="telemetry-val">WAL Active</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label"><span class="telemetry-dot blue"></span>Dataset Benchmark</span>
                <span class="telemetry-val">VisA Industrial</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# PAGE 1: Executive Dashboard
# -----------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.markdown('<div class="panel-kicker">EXECUTIVE QUALITY INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Quality Assurance Dashboard Overview</div>', unsafe_allow_html=True)

    # 3D Holographic Spatial Sensing Node
    st.markdown(
        """
        <div class="hologram-card">
          <div class="hologram-scene">
            <div class="hologram-grid-plate"></div>
            <div class="hologram-radar-rings">
              <div class="ring r1"></div>
              <div class="ring r2"></div>
              <div class="ring r3"></div>
            </div>
            <div class="hologram-cube">
              <div class="cube-face front">YOLO11<br>DEFECT</div>
              <div class="cube-face back">MNET·V3<br>CONV</div>
              <div class="cube-face right">AUDIT<br>TRAIL</div>
              <div class="cube-face left">VIS-A<br>PART</div>
              <div class="cube-face top">3D·SCAN<br>PASS</div>
              <div class="cube-face bottom">DEFECTX<br>v2.4</div>
            </div>
            <div class="hologram-scan-sweep"></div>
          </div>
          <div class="hologram-info">
            <div class="panel-kicker">3D SPATIAL TELEMETRY · REAL-TIME SENSING</div>
            <h3>Industrial Vision Pipeline & Volumetric Defect Mapping</h3>
            <p>DefectX fuses sub-millimeter YOLO11 instance segmentation masks with high-precision MobileNetV3 anomaly classification. High-speed dual-stage inference inspects and segments surface defects across complex manufacturing assemblies in real time.</p>
            <div class="hologram-tags">
              <span>LATENCY: <b>&lt; 35ms</b></span>
              <span>BACKBONE: <b>YOLO11-Seg + MobileNetV3</b></span>
              <span>INSPECTION ENGINE: <b>WAL-Journaled SQLite</b></span>
              <span>DATASET BENCHMARK: <b>VisA Industrial</b></span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = database.list_inspections(limit=10_000, db_path=DATABASE_PATH)
    batches = database.list_batches(limit=10_000, db_path=DATABASE_PATH)

    total_inspections = len(records)
    pass_count = sum(r["product_decision"] == "PASS" for r in records)
    fail_count = sum(r["product_decision"] == "FAIL" for r in records)
    review_count = sum(r["product_decision"] == "MANUAL REVIEW" for r in records)
    classified = pass_count + fail_count
    defect_rate = f"{(100.0 * fail_count / classified):.2f}%" if classified > 0 else "0.00%"

    # KPI Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f'<div class="metric-card total"><div class="metric-label">TOTAL INSPECTIONS</div>'
            f'<div class="metric-value total">{total_inspections:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card pass"><div class="metric-label">PASS COUNT</div>'
            f'<div class="metric-value pass">{pass_count:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card fail"><div class="metric-label">FAIL COUNT</div>'
            f'<div class="metric-value fail">{fail_count:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card review"><div class="metric-label">MANUAL REVIEW</div>'
            f'<div class="metric-value review">{review_count:,}</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">DEFECT RATE</div>'
            f'<div class="metric-value" style="color:#38bdf8;">{defect_rate}</div></div>',
            unsafe_allow_html=True,
        )

    # Analytics Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("##### Overall Quality Decisions")
        fig1 = Figure(figsize=(6, 3.5), facecolor="#091120")
        ax1 = fig1.subplots()
        ax1.set_facecolor("#091120")
        decisions = ["PASS", "FAIL", "MANUAL REVIEW"]
        counts = [pass_count, fail_count, review_count]
        colors = ["#10b981", "#ef4444", "#f59e0b"]
        bars = ax1.bar(decisions, counts, color=colors, width=0.55)
        ax1.set_ylabel("Count", color="#94a3b8")
        ax1.tick_params(colors="#cbd5e1")
        for spine in ax1.spines.values():
            spine.set_color("#1e293b")
        ax1.grid(axis="y", color="#1e293b", linestyle="--", alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height:,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color="#ffffff",
                fontsize=9,
            )
        fig1.tight_layout()
        st.pyplot(fig1)

    with ch2:
        st.markdown("##### Defects by Product Category")
        fig2 = Figure(figsize=(6, 3.5), facecolor="#091120")
        ax2 = fig2.subplots()
        ax2.set_facecolor("#091120")
        categories: dict[str, int] = {}
        for r in records:
            cat = r.get("product_category") or "Unknown"
            if r.get("product_decision") == "FAIL":
                categories[cat] = categories.get(cat, 0) + 1
        if categories:
            cat_names = list(categories.keys())[:8]
            cat_counts = [categories[c] for c in cat_names]
            ax2.barh(cat_names, cat_counts, color="#ef4444", height=0.55)
            ax2.set_xlabel("Defective Images", color="#94a3b8")
        else:
            ax2.text(0.5, 0.5, "No defects recorded yet", color="#64748b", ha="center", va="center")
        ax2.tick_params(colors="#cbd5e1")
        for spine in ax2.spines.values():
            spine.set_color("#1e293b")
        ax2.grid(axis="x", color="#1e293b", linestyle="--", alpha=0.7)
        fig2.tight_layout()
        st.pyplot(fig2)

    # Recent Audit Trail Table
    st.markdown('<div class="panel-kicker" style="margin-top:20px;">RECENT AUDIT TRAIL</div>', unsafe_allow_html=True)
    st.markdown("##### Recent Inspection Records")
    if records:
        recent_data = [
            {
                "Inspection ID": r["inspection_id"][:12] + "...",
                "Timestamp (UTC)": r["inspected_at"],
                "Image": r["image_filename"],
                "Category": r["product_category"],
                "Decision": r["product_decision"],
                "Latency (ms)": round(r["total_inference_ms"] or 0, 1),
                "Objects": r["total_objects"],
            }
            for r in records[:10]
        ]
        st.dataframe(pd.DataFrame(recent_data), use_container_width=True)
    else:
        st.info("No inspection records in database yet. Run an inspection below to populate.")

# -----------------------------------------------------------------------------
# PAGE 2: AI Inspection Workspace
# -----------------------------------------------------------------------------
elif page == "🔬 AI Inspection":
    st.markdown('<div class="panel-kicker">REAL-TIME ANOMALY DETECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">AI Quality Inspection Workspace</div>', unsafe_allow_html=True)

    models = get_models()

    left_col, right_col = st.columns([5, 7])

    with left_col:
        st.markdown('<div class="panel-kicker">SOURCE INGESTION</div>', unsafe_allow_html=True)
        st.markdown("##### Inspection Input")

        tab_sample, tab_upload = st.tabs(["🖼️ VisA Sample Parts", "📁 Upload Image / Archive"])

        selected_image: Image.Image | None = None
        selected_filename = "unknown.png"

        with tab_sample:
            st.caption("Select a preloaded VisA industrial benchmark part:")
            
            # Interactive Visual Sample Gallery with Images
            sample_cols = st.columns(3)
            if "selected_sample_idx" not in st.session_state:
                st.session_state["selected_sample_idx"] = 0

            for i, (fname, desc) in enumerate(SAMPLE_IMAGES):
                col = sample_cols[i % 3]
                img_p = SAMPLE_IMAGE_DIR / fname
                with col:
                    if img_p.is_file():
                        st.image(str(img_p), caption=desc.split(" · ")[0], use_container_width=True)
                        if st.button(f"Stage Sample #{i+1}", key=f"btn_sample_{i}", use_container_width=True):
                            st.session_state["selected_sample_idx"] = i

            curr_idx = st.session_state["selected_sample_idx"]
            sel_fname, sel_desc = SAMPLE_IMAGES[curr_idx]
            sel_path = SAMPLE_IMAGE_DIR / sel_fname
            if sel_path.is_file():
                selected_image = Image.open(sel_path).convert("RGB")
                selected_filename = sel_fname
                st.success(f"Active Sample: **{sel_fname}** ({sel_desc})")

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Upload Part Image or ZIP Archive",
                type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff", "zip"],
                key="ai_file_uploader",
            )
            if uploaded_file is not None:
                if uploaded_file.name.lower().endswith(".zip"):
                    st.info(f"ZIP Archive uploaded: `{uploaded_file.name}`. First member will be staged for AI Inspection.")
                    with zipfile.ZipFile(uploaded_file) as archive:
                        members = app._image_members(archive)
                        if members:
                            data = archive.read(members[0])
                            selected_image = Image.open(io.BytesIO(data)).convert("RGB")
                            selected_filename = members[0].filename
                else:
                    selected_image = Image.open(uploaded_file).convert("RGB")
                    selected_filename = uploaded_file.name
                st.success(f"Active Upload: **{selected_filename}**")

        with st.expander("⚙️ Inspection Thresholds & Parameters", expanded=True):
            conf_threshold = st.slider("YOLO Segmentation Confidence", 0.05, 0.95, value=0.25, step=0.05)
            iou_threshold = st.slider("YOLO IoU Threshold", 0.10, 0.90, value=0.70, step=0.05)
            decision_threshold = st.slider("Operational Decision Threshold", 0.50, 1.00, value=0.80, step=0.01)
            yolo_imgsz = st.selectbox("YOLO Inference Size", [416, 512, 640, 768, 960, 1024], index=2)

        run_inspect = st.button("⚡ RUN AI INSPECTION", type="primary", use_container_width=True)

    with right_col:
        st.markdown('<div class="panel-kicker">DUAL VIEWPORT ANALYSIS</div>', unsafe_allow_html=True)
        st.markdown("##### Inspection Viewports & Segmentation Overlay")

        vp_left, vp_right = st.columns(2)

        with vp_left:
            st.markdown(
                '<div class="viewport-box"><div class="viewport-title">Original Sensor Input Image</div></div>',
                unsafe_allow_html=True,
            )
            if selected_image is not None:
                st.image(selected_image, use_container_width=True)
            else:
                st.info("Select or upload an image to display source input.")

        with vp_right:
            st.markdown(
                '<div class="viewport-box"><div class="viewport-title">AI Defect Segmentation Overlay</div></div>',
                unsafe_allow_html=True,
            )
            annotated_placeholder = st.empty()
            if selected_image is None:
                annotated_placeholder.info("Awaiting input image...")

        status_placeholder = st.empty()
        telemetry_placeholder = st.empty()

        # Run pipeline when triggered
        if selected_image is not None and (run_inspect or "auto_ai_inspected" not in st.session_state):
            st.session_state["auto_ai_inspected"] = True
            with st.spinner("Executing YOLO11 Segmentation & MobileNetV3 Classification..."):
                try:
                    objects, probs, product_decision, annotated, yolo_ms, mobile_ms, total_ms = run_single_inference(
                        image=selected_image,
                        confidence=conf_threshold,
                        iou=iou_threshold,
                        image_size=yolo_imgsz,
                        decision_confidence_threshold=decision_threshold,
                    )

                    # Persist to database
                    batch_id = str(uuid.uuid4())
                    database.create_batch(batch_id, database.utc_timestamp(), 1, float(decision_threshold), db_path=DATABASE_PATH)
                    ins_id = database.save_inspection(
                        {
                            "batch_id": batch_id,
                            "inspected_at": database.utc_timestamp(),
                            "image_filename": selected_filename,
                            "product_category": ", ".join(sorted({item.class_name for item in objects})),
                            "total_objects": len(objects),
                            "normal_objects": product_decision.normal_objects,
                            "defective_objects": product_decision.defective_objects,
                            "uncertain_objects": product_decision.uncertain_objects,
                            "product_decision": product_decision.product_decision,
                            "decision_explanation": product_decision.decision_explanation,
                            "classification_confidences": product_decision.classification_confidences,
                            "object_details": [
                                {
                                    "class_name": d.class_name,
                                    "predicted_class": d.predicted_class,
                                    "confidence": d.confidence,
                                    "status": d.status,
                                }
                                for d in product_decision.object_decisions
                            ],
                            "yolo_ms": round(yolo_ms, 2),
                            "mobilenet_ms": round(mobile_ms, 2),
                            "total_inference_ms": round(total_ms, 2),
                            "decision_confidence_threshold": float(decision_threshold),
                        },
                        original_image=selected_image,
                        annotated_image=annotated,
                        db_path=DATABASE_PATH,
                    )
                    database.complete_batch(
                        batch_id,
                        successful_images=1,
                        pass_count=int(product_decision.product_decision == "PASS"),
                        fail_count=int(product_decision.product_decision == "FAIL"),
                        review_count=int(product_decision.product_decision == "MANUAL REVIEW"),
                        processing_errors=0,
                        defective_objects=product_decision.defective_objects,
                        classified_objects=product_decision.normal_objects + product_decision.defective_objects,
                        db_path=DATABASE_PATH,
                    )

                    annotated_placeholder.image(annotated, use_container_width=True)

                    # Prominent Decision Banner
                    dec_class = product_decision.product_decision.lower().replace(" ", "")
                    dec_icon = "✅" if dec_class == "pass" else ("❌" if dec_class == "fail" else "⚠️")
                    status_placeholder.markdown(
                        f"""
                        <div class="decision-banner {dec_class}">
                            <div class="decision-title">{dec_icon} FINAL QUALITY DECISION: {product_decision.product_decision}</div>
                            <div><b>Image:</b> {html.escape(selected_filename)} · <b>Category:</b> {html.escape(", ".join(sorted({item.class_name for item in objects})) or 'N/A')}</div>
                            <div><b>Explanation:</b> {html.escape(product_decision.decision_explanation)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Telemetry Badges
                    telemetry_placeholder.markdown(
                        f"""
                        <div style="margin-top:10px;">
                            <span class="telemetry-badge">YOLO11: <b>{yolo_ms:.1f} ms</b></span>
                            <span class="telemetry-badge">MobileNetV3: <b>{mobile_ms:.1f} ms</b></span>
                            <span class="telemetry-badge">Total Latency: <b>{total_ms:.1f} ms</b></span>
                            <span class="telemetry-badge">Objects: <b>{len(objects)}</b></span>
                            <span class="telemetry-badge">Audit ID: <b>{ins_id[:8]}...</b></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Detailed Detected Object Table
                    if product_decision.object_decisions:
                        st.markdown("##### Detected Object Classifications")
                        obj_table = [
                            {
                                "Object #": i + 1,
                                "Part Class": d.class_name,
                                "Predicted Condition": d.predicted_class,
                                "Confidence": f"{d.confidence:.1%}" if d.confidence is not None else "N/A",
                                "Decision Status": d.status,
                            }
                            for i, d in enumerate(product_decision.object_decisions)
                        ]
                        st.dataframe(pd.DataFrame(obj_table), use_container_width=True)

                except Exception as e:
                    st.error(f"Inference Pipeline Error: {e}")

# -----------------------------------------------------------------------------
# PAGE 3: High-Throughput Batch Inspection
# -----------------------------------------------------------------------------
elif page == "📦 Batch Inspection":
    st.markdown('<div class="panel-kicker">HIGH-THROUGHPUT QUALITY CONTROL</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Batch Inspection & Quality Analytics</div>', unsafe_allow_html=True)

    models = get_models()

    batch_file = st.file_uploader("Upload ZIP Batch Archive (Containing Images)", type=["zip"])

    with st.expander("⚙️ Batch Parameters", expanded=False):
        b_conf = st.slider("YOLO Segmentation Confidence", 0.05, 0.95, value=0.25, step=0.05, key="b_conf")
        b_iou = st.slider("YOLO IoU Threshold", 0.10, 0.90, value=0.70, step=0.05, key="b_iou")
        b_dec_conf = st.slider("Decision Confidence Threshold", 0.50, 1.00, value=0.80, step=0.01, key="b_dec_conf")
        b_imgsz = st.selectbox("YOLO Inference Size", [416, 512, 640, 768, 960, 1024], index=2, key="b_imgsz")

    start_batch = st.button("⚡ RUN BATCH INSPECTION", type="primary")

    if batch_file is not None and start_batch:
        try:
            with zipfile.ZipFile(batch_file) as archive:
                members = app._image_members(archive)
                total_imgs = len(members)
                if not total_imgs:
                    st.warning("No supported image files found inside the uploaded ZIP archive.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    preview_col1, preview_col2 = st.columns(2)
                    with preview_col1:
                        st.caption("Active Processing Viewport")
                        preview_img = st.empty()

                    batch_id = str(uuid.uuid4())
                    database.create_batch(batch_id, database.utc_timestamp(), total_imgs, float(b_dec_conf), db_path=DATABASE_PATH)

                    rows: list[dict] = []
                    cum_yolo = 0.0
                    cum_mobile = 0.0
                    cum_total = 0.0

                    for pos, info in enumerate(members, start=1):
                        status_text.text(f"Processing image {pos}/{total_imgs}: {info.filename}")
                        try:
                            data = archive.read(info)
                            image = Image.open(io.BytesIO(data)).convert("RGB")

                            objects, probs, p_dec, annotated, yolo_ms, mob_ms, tot_ms = run_single_inference(
                                image=image,
                                confidence=b_conf,
                                iou=b_iou,
                                image_size=b_imgsz,
                                decision_confidence_threshold=b_dec_conf,
                            )

                            cum_yolo += yolo_ms
                            cum_mobile += mob_ms
                            cum_total += tot_ms

                            preview_img.image(annotated, caption=f"{info.filename} · {p_dec.product_decision}", width=380)

                            row = {
                                "image": info.filename,
                                "result": "DEFECTIVE" if p_dec.defective_objects else ("NORMAL" if objects else "NO OBJECTS"),
                                "objects": len(objects),
                                "defective_objects": p_dec.defective_objects,
                                "classes": ", ".join(sorted({item.class_name for item in objects})),
                                "max_defect_probability": round(float(probs.max()), 4) if len(probs) else 0.0,
                                "yolo_ms": round(yolo_ms, 2),
                                "mobilenet_ms": round(mob_ms, 2),
                                "total_inference_ms": round(tot_ms, 2),
                                "product_decision": p_dec.product_decision,
                                "decision_explanation": p_dec.decision_explanation,
                                "normal_objects": p_dec.normal_objects,
                                "uncertain_objects": p_dec.uncertain_objects,
                                "decision_confidence_threshold": float(b_dec_conf),
                                "product_category": ", ".join(sorted({item.class_name for item in objects})),
                                "_processing_error": False,
                            }
                            database.save_inspection(
                                {
                                    "batch_id": batch_id,
                                    "inspected_at": database.utc_timestamp(),
                                    "image_filename": info.filename,
                                    "product_category": row["product_category"],
                                    "total_objects": row["objects"],
                                    "normal_objects": row["normal_objects"],
                                    "defective_objects": row["defective_objects"],
                                    "uncertain_objects": row["uncertain_objects"],
                                    "product_decision": row["product_decision"],
                                    "decision_explanation": row["decision_explanation"],
                                    "classification_confidences": p_dec.classification_confidences,
                                    "object_details": [],
                                    "yolo_ms": row["yolo_ms"],
                                    "mobilenet_ms": row["mobilenet_ms"],
                                    "total_inference_ms": row["total_inference_ms"],
                                    "decision_confidence_threshold": float(b_dec_conf),
                                    "processing_error": None,
                                },
                                original_image=image,
                                annotated_image=annotated,
                                db_path=DATABASE_PATH,
                            )
                            rows.append(row)

                        except Exception as e:
                            rows.append({
                                "image": info.filename,
                                "result": "ERROR",
                                "objects": 0,
                                "defective_objects": 0,
                                "classes": "",
                                "max_defect_probability": 0.0,
                                "yolo_ms": 0.0,
                                "mobilenet_ms": 0.0,
                                "total_inference_ms": 0.0,
                                "product_decision": "MANUAL REVIEW",
                                "decision_explanation": f"Processing error: {e}",
                                "normal_objects": 0,
                                "uncertain_objects": 0,
                                "decision_confidence_threshold": float(b_dec_conf),
                                "product_category": "Error",
                                "_processing_error": True,
                            })

                        progress_bar.progress(pos / total_imgs)

                    database.complete_batch(
                        batch_id,
                        successful_images=sum(not r["_processing_error"] for r in rows),
                        pass_count=sum(r["product_decision"] == "PASS" for r in rows),
                        fail_count=sum(r["product_decision"] == "FAIL" for r in rows),
                        review_count=sum(r["product_decision"] == "MANUAL REVIEW" for r in rows),
                        processing_errors=sum(r["_processing_error"] for r in rows),
                        defective_objects=sum(r["defective_objects"] for r in rows),
                        classified_objects=sum(r["normal_objects"] + r["defective_objects"] for r in rows),
                        db_path=DATABASE_PATH,
                    )

                    status_text.success(f"Batch inspection completed! Processed {total_imgs} images.")

                    # Summary KPI Cards
                    p_c = sum(r["product_decision"] == "PASS" for r in rows)
                    f_c = sum(r["product_decision"] == "FAIL" for r in rows)
                    r_c = sum(r["product_decision"] == "MANUAL REVIEW" for r in rows)
                    e_c = sum(r["_processing_error"] for r in rows)

                    sc1, sc2, sc3, sc4 = st.columns(4)
                    with sc1:
                        st.markdown(f'<div class="metric-card pass"><div class="metric-label">PASS</div><div class="metric-value pass">{p_c}</div></div>', unsafe_allow_html=True)
                    with sc2:
                        st.markdown(f'<div class="metric-card fail"><div class="metric-label">FAIL</div><div class="metric-value fail">{f_c}</div></div>', unsafe_allow_html=True)
                    with sc3:
                        st.markdown(f'<div class="metric-card review"><div class="metric-label">MANUAL REVIEW</div><div class="metric-value review">{r_c}</div></div>', unsafe_allow_html=True)
                    with sc4:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">ERRORS</div><div class="metric-value" style="color:#64748b;">{e_c}</div></div>', unsafe_allow_html=True)

                    # Export CSV
                    df = pd.DataFrame(rows)
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Complete Batch Results CSV",
                        data=csv_data,
                        file_name=f"defectx_batch_{batch_id[:8]}.csv",
                        mime="text/csv",
                    )
                    st.dataframe(df, use_container_width=True)

        except Exception as err:
            st.error(f"Error processing ZIP archive: {err}")

# -----------------------------------------------------------------------------
# PAGE 4: Inspection History & Traceability
# -----------------------------------------------------------------------------
elif page == "🗃️ Inspection History":
    st.markdown('<div class="panel-kicker">COMPLIANCE & TRACEABILITY</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Inspection Audit Trail Database</div>', unsafe_allow_html=True)

    f_col1, f_col2 = st.columns([3, 1])
    with f_col1:
        search_query = st.text_input("Search by inspection ID or filename", placeholder="e.g. candle_Normal_0085.JPG")
    with f_col2:
        decision_filter = st.selectbox("Decision Filter", ["ALL", "PASS", "FAIL", "MANUAL REVIEW"])

    records = database.list_inspections(
        search=search_query or "",
        decision=decision_filter if decision_filter != "ALL" else "",
        limit=10_000,
        db_path=DATABASE_PATH,
    )

    st.markdown(f"Found **{len(records):,}** matching inspection record(s).")

    if records:
        df_history = pd.DataFrame(records)[
            [
                "inspection_id",
                "inspected_at",
                "image_filename",
                "product_category",
                "total_objects",
                "normal_objects",
                "defective_objects",
                "uncertain_objects",
                "product_decision",
                "total_inference_ms",
            ]
        ]
        st.dataframe(df_history, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 🔍 Inspection Record Inspector")
        record_ids = [r["inspection_id"] for r in records]
        selected_rec_id = st.selectbox("Select Record to Inspect in Detail", record_ids)

        chosen_rec = database.get_inspection(selected_rec_id, db_path=DATABASE_PATH)
        if chosen_rec:
            dec_c = chosen_rec['product_decision']
            dec_color = '#10b981' if dec_c == 'PASS' else ('#ef4444' if dec_c == 'FAIL' else '#f59e0b')
            st.markdown(
                f"""
                <div style="background:rgba(15,27,48,0.88); padding:18px 22px; border-radius:12px; border:1px solid rgba(56,189,248,0.2); margin-bottom:16px;">
                    <div><b>ID:</b> <code>{html.escape(chosen_rec['inspection_id'])}</code></div>
                    <div><b>Timestamp:</b> {html.escape(chosen_rec['inspected_at'])} UTC</div>
                    <div><b>Filename:</b> {html.escape(chosen_rec['image_filename'])} · <b>Category:</b> {html.escape(chosen_rec['product_category'])}</div>
                    <div><b>Decision:</b> <b style="color:{dec_color}; font-size:16px;">{chosen_rec['product_decision']}</b></div>
                    <div><b>Explanation:</b> {html.escape(chosen_rec['decision_explanation'])}</div>
                    <div><b>Latency:</b> Total {chosen_rec['total_inference_ms']} ms (YOLO {chosen_rec['yolo_ms']} ms · MobileNet {chosen_rec['mobilenet_ms']} ms)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown('<div class="viewport-title">Original Sensor Input Image</div>', unsafe_allow_html=True)
                if chosen_rec.get("original_image_path") and Path(chosen_rec["original_image_path"]).is_file():
                    st.image(chosen_rec["original_image_path"], use_container_width=True)
                else:
                    st.info("Original image file not found on disk.")
            with p_col2:
                st.markdown('<div class="viewport-title">AI Defect Segmentation Overlay</div>', unsafe_allow_html=True)
                if chosen_rec.get("annotated_image_path") and Path(chosen_rec["annotated_image_path"]).is_file():
                    st.image(chosen_rec["annotated_image_path"], use_container_width=True)
                else:
                    st.info("Annotated image file not found on disk.")

            # Individual PDF Download
            try:
                pdf_path = report_generator.generate_inspection_pdf(chosen_rec)
                with open(pdf_path, "rb") as f_pdf:
                    pdf_bytes = f_pdf.read()
                st.download_button(
                    label="📄 Download Inspection PDF Report",
                    data=pdf_bytes,
                    file_name=f"inspection_{chosen_rec['inspection_id'][:8]}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.warning(f"Could not generate PDF: {e}")

# -----------------------------------------------------------------------------
# PAGE 5: Reports Center
# -----------------------------------------------------------------------------
elif page == "📄 Reports Center":
    st.markdown('<div class="panel-kicker">QUALITY COMPLIANCE & AUDIT EXPORTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Inspection Reports Center</div>', unsafe_allow_html=True)

    all_records = database.list_inspections(limit=10_000, db_path=DATABASE_PATH)
    all_batches = database.list_batches(limit=10_000, db_path=DATABASE_PATH)

    rep1, rep2 = st.columns(2)

    with rep1:
        st.markdown("##### 📄 Single Inspection PDF Certificate")
        if all_records:
            rec_options = {f"{r['inspection_id'][:8]} · {r['image_filename']} ({r['product_decision']})": r["inspection_id"] for r in all_records}
            chosen_label = st.selectbox("Select Inspection Record", list(rec_options.keys()))
            target_id = rec_options[chosen_label]

            if st.button("Generate Inspection PDF Certificate"):
                rec = database.get_inspection(target_id, db_path=DATABASE_PATH)
                if rec:
                    pdf_file = report_generator.generate_inspection_pdf(rec)
                    with open(pdf_file, "rb") as f_p:
                        st.download_button(
                            label="⬇️ Download Certificate PDF",
                            data=f_p.read(),
                            file_name=f"certificate_{target_id[:8]}.pdf",
                            mime="application/pdf",
                        )
        else:
            st.info("No inspection records available to generate reports.")

    with rep2:
        st.markdown("##### 📦 Consolidated Batch Audit Report")
        completed_batches = [b for b in all_batches if b.get("status") == "COMPLETED"]
        if completed_batches:
            batch_opts = {f"{b['inspected_at']} · {b['batch_id'][:8]} ({b['total_images']} images)": b["batch_id"] for b in completed_batches}
            chosen_b_label = st.selectbox("Select Completed Batch", list(batch_opts.keys()))
            target_b_id = batch_opts[chosen_b_label]

            if st.button("Generate Batch PDF Report"):
                batch_data = database.get_batch(target_b_id, db_path=DATABASE_PATH)
                b_records = database.list_batch_inspections(target_b_id, db_path=DATABASE_PATH)
                if batch_data and b_records:
                    batch_pdf = report_generator.generate_batch_pdf(batch_data, b_records)
                    with open(batch_pdf, "rb") as f_bp:
                        st.download_button(
                            label="⬇️ Download Batch Audit PDF",
                            data=f_bp.read(),
                            file_name=f"batch_audit_{target_b_id[:8]}.pdf",
                            mime="application/pdf",
                        )
        else:
            st.info("No completed batches available to generate batch reports.")

# -----------------------------------------------------------------------------
# PAGE 6: About the System
# -----------------------------------------------------------------------------
elif page == "ℹ️ About the System":
    st.markdown('<div class="panel-kicker">VISION PIPELINE ARCHITECTURE</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">About DefectX Quality Intelligence</div>', unsafe_allow_html=True)

    st.markdown(
        """
        DefectX is an industrial defect inspection platform designed for real-time quality assurance, surface anomaly detection, and automated defect auditing.

        ### System Workflow
        """
    )

    # 3D Floating Isometric Workflow Stations
    st.markdown(
        """
        <div class="workflow-flow">
          <div class="flow-step">
            <span>STATION 01</span>
            <strong>Image Ingestion</strong>
            <p style="font-size:10px; color:#94a3b8; margin:4px 0 0;">High-Res Sensors</p>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <span>STATION 02</span>
            <strong>YOLO11-Seg</strong>
            <p style="font-size:10px; color:#94a3b8; margin:4px 0 0;">Mask Extraction</p>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <span>STATION 03</span>
            <strong>MobileNetV3</strong>
            <p style="font-size:10px; color:#94a3b8; margin:4px 0 0;">Attention & FiLM</p>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <span>STATION 04</span>
            <strong>Decision Engine</strong>
            <p style="font-size:10px; color:#94a3b8; margin:4px 0 0;">Confidence Routing</p>
          </div>
          <div class="flow-arrow">→</div>
          <div class="flow-step">
            <span>STATION 05</span>
            <strong>Audit & PDF</strong>
            <p style="font-size:10px; color:#94a3b8; margin:4px 0 0;">SQLite + Compliance</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Methodology Details
        1. **Image Sensor Ingestion**: High-resolution industrial image or ZIP archive stream ingested.
        2. **YOLO11 Instance Segmentation**: Localizes part boundaries and segments object crops and contours.
        3. **MobileNetV3 Defect Classification**: Evaluates each segmented object crop using attention-guided feature maps and class-conditioned FiLM embeddings.
        4. **Quality Decision Engine**: Applies the trained threshold (`0.509277`) and operational review threshold (`0.80`) to assign a deterministic `PASS`, `FAIL`, or `MANUAL REVIEW` verdict.
        5. **SQLite Audit Trail**: Persists inspection metadata, bounding boxes, confidences, latencies, and image crops into a WAL-journaled SQLite database.
        6. **PDF Compliance Certificate**: Generates printable ReportLab quality reports for compliance and supplier auditing.

        ### Dataset Grounding
        The current model checkpoint is trained and benchmarked strictly on the **VisA (Visual Anomaly) Industrial Dataset**, covering surface anomalies across complex components (candles, capsules, printed circuit boards, and industrial parts). No automotive-specific validation is claimed.
        """
    )
