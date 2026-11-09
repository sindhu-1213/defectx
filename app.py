#!/usr/bin/env python3
"""DefectX Gradio application: YOLO11 segmentation -> MobileNetV3 classification."""

from __future__ import annotations

import argparse
import html
import io
import os
import tempfile
from pathlib import Path, PurePosixPath

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "defectx_mpl"))
try:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
except Exception:
    pass
import time
import uuid
import zipfile
import zlib
from dataclasses import dataclass

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from matplotlib.figure import Figure
from ultralytics import YOLO

from src.models import build_model, build_transforms
from src.decision_engine import ObjectPrediction, ProductDecision, decide_product
from src import database, report_generator


HERE = Path(__file__).resolve().parent
DATABASE_PATH = database.DEFAULT_DATABASE_PATH
DEFAULT_YOLO = HERE / "models" / "yolo11_seg_best.pt"
DEFAULT_MOBILENET = HERE / "models" / "mobilenet_v3_best.pt"
SAMPLE_IMAGE_DIR = HERE / "assets" / "sample_images"
SAMPLE_IMAGES = [
    ("candle_Normal_0085.JPG", "Candle · source label: normal"),
    ("candle_Anomaly_009.JPG", "Candle · source label: anomaly"),
    ("capsules_Normal_033.JPG", "Capsules · source label: normal"),
    ("capsules_Anomaly_000.JPG", "Capsules · source label: anomaly"),
    ("pcb2_Normal_0041.JPG", "PCB 2 · source label: normal"),
    ("pcb2_Anomaly_001.JPG", "PCB 2 · source label: anomaly"),
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RESULT_COLUMNS = [
    "image",
    "result",
    "objects",
    "defective_objects",
    "classes",
    "max_defect_probability",
    "yolo_ms",
    "mobilenet_ms",
    "total_inference_ms",
    "message",
    "product_decision",
    "decision_explanation",
    "normal_objects",
    "uncertain_objects",
    "decision_confidence_threshold",
    "classification_confidences",
    "product_category",
    "max_classification_confidence",
]
HISTORY_COLUMNS = [
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
MAX_IMAGES = 10_000
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 150_000_000


@dataclass
class SegmentedObject:
    crop: Image.Image
    class_name: str
    class_id: int | None
    yolo_confidence: float
    box: tuple[int, int, int, int]
    polygon: np.ndarray | None


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


class InspectionModels:
    """Models loaded once and reused for every Gradio request."""

    def __init__(self, yolo_path: Path, mobilenet_path: Path):
        for path in (yolo_path, mobilenet_path):
            if not path.is_file():
                raise FileNotFoundError(f"Model checkpoint not found: {path}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(str(yolo_path), task="segment")
        checkpoint = torch.load(mobilenet_path, map_location="cpu", weights_only=False)
        self.class_to_idx = {
            str(name): int(index) for name, index in checkpoint["class_to_idx"].items()
        }
        self.threshold = float(checkpoint["threshold"])
        config = checkpoint.get("config", {})
        self.image_size = int(config.get("image_size", 224))
        self.mobilenet = build_model(
            backbone="mobilenet_v3_large",
            num_classes=len(self.class_to_idx),
            device=self.device,
            pretrained=False,
            embedding_dim=int(config.get("embedding_dim", 128)),
            dropout=float(config.get("dropout", 0.30)),
        )
        self.mobilenet.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.mobilenet.eval()
        _, self.transform = build_transforms(self.image_size)

        yolo_names = _model_names(self.yolo.names)
        yolo_classes = set(yolo_names.values())
        mobile_classes = set(self.class_to_idx)
        if yolo_classes != mobile_classes:
            raise ValueError(
                "YOLO/MobileNet class mismatch. "
                f"Only YOLO: {sorted(yolo_classes - mobile_classes)}; "
                f"only MobileNet: {sorted(mobile_classes - yolo_classes)}"
            )
        self.yolo_names = yolo_names
        self._warm_up()

    def _warm_up(self) -> None:
        """Exclude one-time CUDA/kernel initialization from reported inference time."""
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo.predict(
            source=blank,
            imgsz=640,
            conf=0.25,
            device=str(self.device),
            verbose=False,
        )
        tensor = torch.zeros(
            (1, 3, self.image_size, self.image_size), device=self.device
        ).to(memory_format=torch.channels_last)
        class_id = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            self.mobilenet(tensor, class_id)
        _sync_cuda(self.device)

    def segment(
        self, image: Image.Image, confidence: float, iou: float, image_size: int
    ) -> tuple[list[SegmentedObject], float]:
        rgb = np.asarray(image.convert("RGB"))
        _sync_cuda(self.device)
        started = time.perf_counter()
        result = self.yolo.predict(
            source=rgb,
            imgsz=image_size,
            conf=confidence,
            iou=iou,
            device=str(self.device),
            retina_masks=True,
            verbose=False,
        )[0]
        _sync_cuda(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if result.boxes is None or len(result.boxes) == 0:
            return [], elapsed_ms

        height, width = rgb.shape[:2]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        polygons = result.masks.xy if result.masks is not None else [None] * len(boxes)
        objects: list[SegmentedObject] = []

        for box, class_id, score, polygon in zip(
            boxes, class_ids, confidences, polygons
        ):
            x1, y1, x2, y2 = box
            padding = max(2, int(0.02 * max(x2 - x1, y2 - y1)))
            left = max(0, int(np.floor(x1)) - padding)
            top = max(0, int(np.floor(y1)) - padding)
            right = min(width, int(np.ceil(x2)) + padding)
            bottom = min(height, int(np.ceil(y2)) + padding)
            if right <= left or bottom <= top:
                continue

            polygon_array = None
            object_image = rgb
            if polygon is not None and len(polygon) >= 3:
                polygon_array = np.asarray(polygon, dtype=np.float32)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [np.rint(polygon_array).astype(np.int32)], 255)
                object_image = cv2.bitwise_and(rgb, rgb, mask=mask)
            crop = Image.fromarray(object_image[top:bottom, left:right]).convert("RGB")
            class_name = self.yolo_names.get(int(class_id), f"class_{class_id}")
            objects.append(
                SegmentedObject(
                    crop=crop,
                    class_name=class_name,
                    class_id=self.class_to_idx.get(class_name),
                    yolo_confidence=float(score),
                    box=(left, top, right, bottom),
                    polygon=polygon_array,
                )
            )
        return objects, elapsed_ms

    def classify(
        self, objects: list[SegmentedObject]
    ) -> tuple[np.ndarray, float]:
        probabilities = np.full(len(objects), np.nan, dtype=np.float32)
        classifiable_indices = [
            index for index, item in enumerate(objects) if item.class_id is not None
        ]
        if not classifiable_indices:
            return probabilities, 0.0

        # Preprocessing is included in MobileNet pipeline time; model loading is not.
        _sync_cuda(self.device)
        started = time.perf_counter()
        classifiable_objects = [objects[index] for index in classifiable_indices]
        images = torch.stack(
            [self.transform(item.crop) for item in classifiable_objects]
        ).to(self.device, non_blocking=True, memory_format=torch.channels_last)
        class_ids = torch.tensor(
            [item.class_id for item in classifiable_objects],
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            logits, _ = self.mobilenet(images, class_ids)
            predicted_probabilities = torch.sigmoid(logits)
        _sync_cuda(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        probabilities[classifiable_indices] = predicted_probabilities.float().cpu().numpy()
        return probabilities, elapsed_ms


MODELS: InspectionModels | None = None


def annotate(
    image: Image.Image,
    objects: list[SegmentedObject],
    probabilities: np.ndarray,
    classifier_threshold: float,
    decision_confidence_threshold: float,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_width = max(2, round(max(canvas.size) / 500))

    for index, item in enumerate(objects):
        probability = (
            float(probabilities[index]) if index < len(probabilities) else float("nan")
        )
        if item.class_id is None or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            color = (217, 119, 6)
            label = f"{index + 1} {item.class_name} | MANUAL REVIEW"
        else:
            defective = probability >= classifier_threshold
            classification = "DEFECTIVE" if defective else "NORMAL"
            confidence = probability if defective else 1.0 - probability
            if confidence < decision_confidence_threshold:
                color = (217, 119, 6)
                label = f"{index + 1} {item.class_name} | REVIEW {classification} {confidence:.1%}"
            else:
                color = (225, 45, 45) if defective else (35, 180, 90)
                label = f"{index + 1} {item.class_name} | {classification} {confidence:.1%}"
        if item.polygon is not None:
            points = [tuple(map(float, point)) for point in item.polygon]
            draw.line(points + [points[0]], fill=color, width=line_width)
        draw.rectangle(item.box, outline=color, width=line_width)
        text_box = draw.textbbox((item.box[0], item.box[1]), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        tx = item.box[0]
        ty = max(0, item.box[1] - text_height - 6)
        draw.rectangle((tx, ty, tx + text_width + 6, ty + text_height + 6), fill=color)
        draw.text((tx + 3, ty + 3), label, fill="white", font=font)
    return canvas


def _zip_path(upload) -> Path:
    if upload is None:
        raise gr.Error("Upload a ZIP file or select a sample image first.")
    if isinstance(upload, (list, tuple)) and upload:
        upload = upload[0]
    if isinstance(upload, dict):
        value = upload.get("path") or upload.get("name")
    elif isinstance(upload, (str, os.PathLike)):
        value = upload
    elif hasattr(upload, "path") and getattr(upload, "path"):
        value = upload.path
    elif hasattr(upload, "name") and getattr(upload, "name"):
        value = upload.name
    else:
        value = str(upload)
    if not value:
        raise gr.Error("The provided file reference is invalid.")
    path = Path(value)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        with tempfile.NamedTemporaryFile(
            prefix="defectx_single_", suffix=".zip", delete=False
        ) as handle:
            archive_path = Path(handle.name)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(path, arcname=path.name)
        return archive_path
    if path.suffix.lower() != ".zip" or not zipfile.is_zipfile(path):
        raise gr.Error("The uploaded file is not a valid ZIP archive.")
    return path


def _image_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    total_bytes = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or path.name.startswith(".") or "__MACOSX" in path.parts:
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if info.file_size > MAX_IMAGE_BYTES:
            raise gr.Error(f"Image is too large: {info.filename}")
        total_bytes += info.file_size
        if total_bytes > MAX_TOTAL_BYTES:
            raise gr.Error("Uncompressed image data exceeds the 4 GiB safety limit.")
        members.append(info)
    members.sort(key=lambda item: item.filename.casefold())
    if len(members) > MAX_IMAGES:
        raise gr.Error(f"ZIP contains {len(members):,} images; limit is {MAX_IMAGES:,}.")
    return members


def _timing_markdown(
    processed: int,
    total: int,
    yolo_ms: float,
    mobile_ms: float,
    inference_ms: float,
) -> str:
    average = inference_ms / processed if processed else 0.0
    throughput = 1000.0 / average if average else 0.0
    return (
        "### Inference time (model loading excluded)\n"
        f"**Processed:** {processed:,}/{total:,} images  \n"
        f"**YOLO11:** {yolo_ms / 1000:.3f} s  \n"
        f"**MobileNetV3:** {mobile_ms / 1000:.3f} s  \n"
        f"**Combined inference:** {inference_ms / 1000:.3f} s  \n"
        f"**Average:** {average:.1f} ms/image · **Throughput:** {throughput:.2f} images/s"
    )


def _decision_status_html(
    progress: str,
    filename: str,
    decision: ProductDecision,
    confidence_threshold: float,
) -> str:
    colors = {
        "PASS": ("#15803d", "#f0fdf4"),
        "FAIL": ("#b91c1c", "#fef2f2"),
        "MANUAL REVIEW": ("#b45309", "#fffbeb"),
    }
    foreground, background = colors[decision.product_decision]
    return (
        f'<section class="decision-result" style="border-left:5px solid {foreground};'
        f'background:linear-gradient(120deg,{background}22,rgba(13,25,45,.92));'
        'padding:18px 22px;border-radius:16px;color:#e5f1ff;'
        'box-shadow:0 12px 35px rgba(0,0,0,.2)">'
        f'<div style="font-size:12px;color:#8da3c3;letter-spacing:.08em;text-transform:uppercase">'
        f'{html.escape(progress)} · '
        f'{html.escape(filename)}</div>'
        f'<strong style="display:block;color:{foreground};font-size:30px;'
        'letter-spacing:.06em;margin:6px 0">'
        f'{html.escape(decision.product_decision)}</strong>'
        f'<div style="color:#d7e6fb">Objects: '
        f'{decision.normal_objects + decision.defective_objects + decision.uncertain_objects} total · '
        f'{decision.normal_objects} normal · {decision.defective_objects} defective · '
        f'{decision.uncertain_objects} uncertain</div>'
        f'<div style="color:#91a7c7;margin-top:5px">Operational decision threshold: '
        f'{confidence_threshold:.2f} · not a calibrated probability</div>'
        f'<div style="margin-top:10px;color:#edf5ff">'
        f'{html.escape(decision.decision_explanation)}</div>'
        f'<div style="margin-top:10px;font-size:12px;color:#9eb4d4;line-height:1.6">'
        f'{html.escape(decision.classification_confidences)}</div>'
        "</section>"
    )


def _filtered_rows(rows: list[dict], decision_filter: str) -> list[list]:
    selected = rows
    if decision_filter != "ALL":
        selected = [
            row for row in rows if row["product_decision"] == decision_filter
        ]
    return [[row.get(column, "") for column in RESULT_COLUMNS] for row in selected]


def filter_batch_results(decision_filter: str, rows: list[dict]) -> list[list]:
    return _filtered_rows(rows or [], decision_filter)


def refresh_inspection_history(
    search_text: str = "", decision_filter: str = "ALL"
) -> tuple[list[list], gr.Dropdown, gr.Dropdown, str]:
    records = database.list_inspections(
        search=search_text or "",
        decision=decision_filter or "ALL",
        limit=10_000,
        db_path=DATABASE_PATH,
    )
    batches = database.list_batches(limit=10_000, db_path=DATABASE_PATH)
    history_rows = [
        [record.get(column) for column in HISTORY_COLUMNS] for record in records
    ]
    inspection_choices = [
        (
            f"{record['inspection_id']} · {record['image_filename']}",
            record["inspection_id"],
        )
        for record in records
    ]
    batch_choices = [
        (
            f"{batch['inspected_at']} · {batch['batch_id']} "
            f"({batch['total_images']} images)",
            batch["batch_id"],
        )
        for batch in batches
    ]
    counts = {
        decision: sum(
            record["product_decision"] == decision for record in records
        )
        for decision in ("PASS", "FAIL", "MANUAL REVIEW")
    }
    summary = (
        f"<p>Showing <strong>{len(records)}</strong> inspection(s) · "
        f'<span style="color:#15803d">PASS {counts["PASS"]}</span> · '
        f'<span style="color:#b91c1c">FAIL {counts["FAIL"]}</span> · '
        f'<span style="color:#b45309">MANUAL REVIEW {counts["MANUAL REVIEW"]}</span>'
        f" · {len(batches)} completed batch(es)</p>"
    )
    return (
        history_rows,
        gr.Dropdown(
            choices=inspection_choices,
            value=inspection_choices[0][1] if inspection_choices else None,
            interactive=bool(inspection_choices),
        ),
        gr.Dropdown(
            choices=batch_choices,
            value=batch_choices[0][1] if batch_choices else None,
            interactive=bool(batch_choices),
        ),
        summary,
    )


def view_inspection_details(inspection_id: str | None) -> tuple[str, Image.Image | None, Image.Image | None]:
    if not inspection_id:
        raise gr.Error("Select an inspection record first.")
    record = database.get_inspection(inspection_id, db_path=DATABASE_PATH)
    if record is None:
        raise gr.Error("The selected inspection record was not found.")

    detail_rows = []
    for index, item in enumerate(record["object_details"], start=1):
        confidence = item.get("confidence")
        confidence_text = (
            "Not available" if confidence is None else f"{confidence:.3f}"
        )
        detail_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(str(item.get('class_name') or 'Unknown'))}</td>"
            f"<td>{html.escape(str(item.get('predicted_class') or 'UNKNOWN'))}</td>"
            f"<td>{confidence_text}</td>"
            f"<td>{html.escape(str(item.get('status') or 'uncertain'))}</td>"
            "</tr>"
        )
    details = "".join(detail_rows)
    if not details:
        details = '<tr><td colspan="5">No detected objects</td></tr>'
    explanation = html.escape(record["decision_explanation"])
    summary = (
        f"<section><h3>{html.escape(record['product_decision'])}</h3>"
        f"<p><strong>ID:</strong> {html.escape(record['inspection_id'])}<br>"
        f"<strong>Time (UTC):</strong> {html.escape(record['inspected_at'])}<br>"
        f"<strong>Image:</strong> {html.escape(record['image_filename'])}<br>"
        f"<strong>Category:</strong> {html.escape(record['product_category'])}<br>"
        f"<strong>Objects:</strong> {record['total_objects']} total · "
        f"{record['normal_objects']} normal · {record['defective_objects']} defective · "
        f"{record['uncertain_objects']} uncertain<br>"
        f"<strong>Inference (ms):</strong> YOLO11 {record['yolo_ms']} · "
        f"MobileNetV3 {record['mobilenet_ms']} · total {record['total_inference_ms']}<br>"
        f"<strong>Decision threshold:</strong> {record['decision_confidence_threshold']:.2f}</p>"
        f"<p><strong>Explanation:</strong> {explanation}</p>"
        "<table style='border-collapse:collapse;width:100%'><thead><tr>"
        "<th>#</th><th>Class</th><th>Classification</th><th>Confidence</th><th>Status</th>"
        f"</tr></thead><tbody>{details}</tbody></table></section>"
    )

    def open_saved_image(path_value: str | None) -> Image.Image | None:
        if not path_value or not Path(path_value).is_file():
            return None
        with Image.open(path_value) as saved:
            return saved.convert("RGB")

    return (
        summary,
        open_saved_image(record.get("original_image_path")),
        open_saved_image(record.get("annotated_image_path")),
    )


def download_inspection_pdf(inspection_id: str | None) -> str:
    if not inspection_id:
        raise gr.Error("Select an inspection record first.")
    record = database.get_inspection(inspection_id, db_path=DATABASE_PATH)
    if record is None:
        raise gr.Error("The selected inspection record was not found.")
    return report_generator.generate_inspection_pdf(record)


def download_batch_pdf(batch_id: str | None) -> str:
    if not batch_id:
        raise gr.Error("Select a completed batch first.")
    batch = database.get_batch(batch_id, db_path=DATABASE_PATH)
    if batch is None or batch["status"] != "COMPLETED":
        raise gr.Error("The selected completed batch was not found.")
    records = database.list_batch_inspections(batch_id, db_path=DATABASE_PATH)
    return report_generator.generate_batch_pdf(batch, records)


def _batch_summary_html(rows: list[dict], total_submitted: int) -> str:
    errors = sum(bool(row.get("_processing_error")) for row in rows)
    successful = len(rows) - errors
    decisions = {
        decision: sum(row["product_decision"] == decision for row in rows)
        for decision in ("PASS", "FAIL", "MANUAL REVIEW")
    }
    normal_objects = sum(row["normal_objects"] for row in rows)
    defective_objects = sum(row["defective_objects"] for row in rows)
    uncertain_objects = sum(row["uncertain_objects"] for row in rows)
    classified_objects = normal_objects + defective_objects
    defect_rate = (
        f"{100 * defective_objects / classified_objects:.2f}%"
        if classified_objects
        else "N/A"
    )
    cards = [
        ("Images submitted", total_submitted, "#334155"),
        ("Successfully processed", successful, "#0369a1"),
        ("PASS", decisions["PASS"], "#15803d"),
        ("FAIL", decisions["FAIL"], "#b91c1c"),
        ("MANUAL REVIEW", decisions["MANUAL REVIEW"], "#b45309"),
        ("Processing errors", errors, "#475569"),
    ]
    card_html = "".join(
        f'<div class="metric-card" style="background:rgba(15,27,48,.84);'
        f'border:1px solid rgba(135,170,220,.16);border-radius:14px;'
        f'padding:16px;min-width:135px;flex:1"><div style="color:#91a7c7;font-size:12px;'
        f'letter-spacing:.08em;text-transform:uppercase">'
        f'{html.escape(str(label))}</div><strong style="display:block;color:{color};'
        f'font-size:28px;margin-top:4px">{value}</strong></div>'
        for label, value, color in cards
    )
    return (
        f'<div class="metric-grid" style="display:flex;flex-wrap:wrap;gap:12px">{card_html}</div>'
        f'<p style="color:#adbed8;margin-top:18px"><strong style="color:#e5f1ff">'
        f'Overall object defect rate:</strong> {defect_rate} — '
        f'{defective_objects} confidently defective objects / '
        f'{classified_objects} confidently classified objects. Uncertain objects '
        "and processing errors are excluded from this denominator.</p>"
        f'<p style="color:#91a7c7">Object totals: {normal_objects} normal · '
        f'{defective_objects} defective · '
        f'{uncertain_objects} uncertain</p>'
    )


def _build_batch_charts(rows: list[dict]) -> tuple[Figure, Figure, Figure, Figure]:
    decision_fig = Figure(figsize=(6, 3.6))
    decision_ax = decision_fig.subplots()
    decision_names = ("PASS", "FAIL", "MANUAL REVIEW")
    decision_counts = [
        sum(row["product_decision"] == name for row in rows)
        for name in decision_names
    ]
    decision_ax.bar(decision_names, decision_counts, color=("#16a34a", "#dc2626", "#d97706"))
    decision_ax.set_title("Image-level quality decisions")
    decision_ax.set_ylabel("Images")
    decision_ax.tick_params(axis="x", rotation=12)

    object_fig = Figure(figsize=(6, 3.6))
    object_ax = object_fig.subplots()
    object_names = ("Normal", "Defective", "Uncertain")
    object_counts = [
        sum(row["normal_objects"] for row in rows),
        sum(row["defective_objects"] for row in rows),
        sum(row["uncertain_objects"] for row in rows),
    ]
    object_ax.bar(object_names, object_counts, color=("#16a34a", "#dc2626", "#d97706"))
    object_ax.set_title("Object classification outcomes")
    object_ax.set_ylabel("Objects")

    category_counts: dict[str, int] = {}
    for row in rows:
        for detail in row.get("_object_details", []):
            if detail["status"] == "defective":
                category_counts[detail["class_name"]] = (
                    category_counts.get(detail["class_name"], 0) + 1
                )
    category_fig = Figure(figsize=(6, 3.6))
    category_ax = category_fig.subplots()
    categories = sorted(category_counts)
    category_ax.bar(
        categories,
        [category_counts[name] for name in categories],
        color="#dc2626",
    )
    category_ax.set_title("Confident defects by product category")
    category_ax.set_ylabel("Defective objects")
    category_ax.tick_params(axis="x", rotation=35)
    if not categories:
        category_ax.text(0.5, 0.5, "No confidently classified defects", ha="center", va="center")

    timing_rows = [
        row for row in rows
        if isinstance(row.get("total_inference_ms"), (int, float))
    ]
    timing_fig = Figure(figsize=(8, 3.6))
    timing_ax = timing_fig.subplots()
    timing_ax.bar(
        [Path(row["image"]).name for row in timing_rows],
        [row["total_inference_ms"] for row in timing_rows],
        color="#2563eb",
    )
    timing_ax.set_title("Per-image inference time")
    timing_ax.set_ylabel("Milliseconds")
    timing_ax.tick_params(axis="x", rotation=45, labelsize=8)
    if not timing_rows:
        timing_ax.text(0.5, 0.5, "No successful inference timings", ha="center", va="center")

    for figure in (decision_fig, object_fig, category_fig, timing_fig):
        figure.patch.set_facecolor("#0b1424")
        for axis in figure.axes:
            axis.set_facecolor("#0b1424")
            axis.tick_params(colors="#b9c9e2")
            axis.xaxis.label.set_color("#91a7c7")
            axis.yaxis.label.set_color("#91a7c7")
            axis.title.set_color("#e5f1ff")
            axis.grid(axis="y", color="#283750", alpha=0.65, linewidth=0.7)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_color("#293952")
        figure.subplots_adjust(bottom=0.28)
    return decision_fig, object_fig, category_fig, timing_fig


def inspect_zip(
    upload,
    yolo_confidence: float,
    yolo_iou: float,
    yolo_image_size: int,
    decision_confidence_threshold: float = 0.80,
    decision_filter: str = "ALL",
):
    if MODELS is None:
        raise gr.Error("Models are not loaded. Restart the application.")

    path = _zip_path(upload)
    rows: list[dict] = []
    cumulative_yolo_ms = 0.0
    cumulative_mobile_ms = 0.0
    cumulative_inference_ms = 0.0

    with zipfile.ZipFile(path) as archive:
        members = _image_members(archive)
        total = len(members)
        batch_id = str(uuid.uuid4())
        batch_timestamp = database.utc_timestamp()
        database.create_batch(
            batch_id,
            batch_timestamp,
            total,
            float(decision_confidence_threshold),
            db_path=DATABASE_PATH,
        )
        yield (
            f"<div>Starting inspection · found <strong>{total:,}</strong> images. Models are already loaded.</div>",
            None,
            [],
            _timing_markdown(0, total, 0.0, 0.0, 0.0),
            None,
            _batch_summary_html([], total),
            None,
            None,
            None,
            None,
            [],
        )

        for position, info in enumerate(members, start=1):
            annotated = None
            image = None
            objects: list[SegmentedObject] = []
            probabilities = np.empty(0, dtype=np.float32)
            yolo_ms = 0.0
            mobile_ms = 0.0
            total_ms = 0.0
            processing_error = None
            try:
                # ZIP reading and image decoding happen before the inference timer.
                data = archive.read(info)
                with Image.open(io.BytesIO(data)) as opened:
                    image = opened.convert("RGB")

                pipeline_started = time.perf_counter()
                objects, yolo_ms = MODELS.segment(
                    image,
                    confidence=float(yolo_confidence),
                    iou=float(yolo_iou),
                    image_size=int(yolo_image_size),
                )
                probabilities, mobile_ms = MODELS.classify(objects)
                _sync_cuda(MODELS.device)
                total_ms = (time.perf_counter() - pipeline_started) * 1000.0

                cumulative_yolo_ms += yolo_ms
                cumulative_mobile_ms += mobile_ms
                cumulative_inference_ms += total_ms
            except (
                UnidentifiedImageError,
                Image.DecompressionBombError,
                OSError,
                ValueError,
                RuntimeError,
                cv2.error,
                zipfile.BadZipFile,
                zlib.error,
                EOFError,
                OverflowError,
                TypeError,
            ) as error:
                processing_error = str(error)

            predictions = [
                ObjectPrediction(
                    class_name=item.class_name,
                    defect_probability=(
                        float(probabilities[index])
                        if index < len(probabilities)
                        else None
                    ),
                )
                for index, item in enumerate(objects)
            ]
            product_decision = decide_product(
                predictions,
                classifier_threshold=MODELS.threshold,
                decision_confidence_threshold=float(decision_confidence_threshold),
                known_classes=set(MODELS.class_to_idx),
                processing_error=processing_error,
            )
            valid_probabilities = probabilities[
                np.isfinite(probabilities) & (probabilities >= 0.0) & (probabilities <= 1.0)
            ]
            predicted_defective_count = int(
                np.sum(valid_probabilities >= MODELS.threshold)
            )
            if processing_error:
                result = "ERROR"
                message = processing_error
            elif not objects:
                result = "NO OBJECTS"
                message = "YOLO11 found no objects at the selected confidence."
            else:
                result = "DEFECTIVE" if predicted_defective_count else "NORMAL"
                message = ""

            max_probability = (
                float(valid_probabilities.max()) if len(valid_probabilities) else None
            )
            if image is not None:
                annotated = annotate(
                    image,
                    objects,
                    probabilities,
                    MODELS.threshold,
                    float(decision_confidence_threshold),
                )
            row = {
                "image": info.filename,
                "result": result,
                "objects": len(objects),
                "defective_objects": product_decision.defective_objects,
                "classes": ", ".join(sorted({item.class_name for item in objects})),
                "max_defect_probability": (
                    "" if max_probability is None else round(max_probability, 6)
                ),
                "yolo_ms": round(yolo_ms, 2) if not processing_error else "",
                "mobilenet_ms": round(mobile_ms, 2) if not processing_error else "",
                "total_inference_ms": round(total_ms, 2) if not processing_error else "",
                "message": message,
                "product_decision": product_decision.product_decision,
                "decision_explanation": product_decision.decision_explanation,
                "normal_objects": product_decision.normal_objects,
                "uncertain_objects": product_decision.uncertain_objects,
                "decision_confidence_threshold": float(decision_confidence_threshold),
                "classification_confidences": product_decision.classification_confidences,
                "product_category": ", ".join(sorted({item.class_name for item in objects})),
                "max_classification_confidence": (
                    ""
                    if product_decision.max_classification_confidence is None
                    else product_decision.max_classification_confidence
                ),
                "_processing_error": processing_error is not None,
                "_object_details": [
                    {
                        "class_name": decision.class_name,
                        "predicted_class": decision.predicted_class,
                        "confidence": decision.confidence,
                        "status": decision.status,
                    }
                    for decision in product_decision.object_decisions
                ],
            }
            row["_inspection_id"] = database.save_inspection(
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
                    "classification_confidences": row["classification_confidences"],
                    "object_details": row["_object_details"],
                    "yolo_ms": row["yolo_ms"],
                    "mobilenet_ms": row["mobilenet_ms"],
                    "total_inference_ms": row["total_inference_ms"],
                    "decision_confidence_threshold": row[
                        "decision_confidence_threshold"
                    ],
                    "processing_error": processing_error,
                },
                original_image=image,
                annotated_image=annotated,
                db_path=DATABASE_PATH,
            )
            rows.append(row)

            status = _decision_status_html(
                f"Image {position:,}/{total:,}",
                info.filename,
                product_decision,
                float(decision_confidence_threshold),
            )
            table = _filtered_rows(rows, decision_filter)
            yield (
                status,
                annotated,
                table,
                _timing_markdown(
                    position,
                    total,
                    cumulative_yolo_ms,
                    cumulative_mobile_ms,
                    cumulative_inference_ms,
                ),
                None,
                _batch_summary_html(rows, total),
                None,
                None,
                None,
                None,
                rows,
            )

    normal_objects = sum(row["normal_objects"] for row in rows)
    defective_objects = sum(row["defective_objects"] for row in rows)
    database.complete_batch(
        batch_id,
        successful_images=sum(not row["_processing_error"] for row in rows),
        pass_count=sum(row["product_decision"] == "PASS" for row in rows),
        fail_count=sum(row["product_decision"] == "FAIL" for row in rows),
        review_count=sum(
            row["product_decision"] == "MANUAL REVIEW" for row in rows
        ),
        processing_errors=sum(row["_processing_error"] for row in rows),
        defective_objects=defective_objects,
        classified_objects=normal_objects + defective_objects,
        db_path=DATABASE_PATH,
    )
    report = pd.DataFrame(
        [{column: row.get(column, "") for column in RESULT_COLUMNS} for row in rows],
        columns=RESULT_COLUMNS,
    )
    with tempfile.NamedTemporaryFile(
        prefix="inspection_results_", suffix=".csv", delete=False
    ) as handle:
        report_path = Path(handle.name)
    report.to_csv(report_path, index=False)
    if rows:
        latest = rows[-1]
        final_decision = ProductDecision(
            product_decision=latest["product_decision"],
            decision_explanation=latest["decision_explanation"],
            normal_objects=latest["normal_objects"],
            defective_objects=latest["defective_objects"],
            uncertain_objects=latest["uncertain_objects"],
            classification_confidences=latest["classification_confidences"],
            max_classification_confidence=latest["max_classification_confidence"] or None,
            object_decisions=(),
        )
        final_status = _decision_status_html(
            f"Batch complete · {len(report):,} image(s)",
            latest["image"],
            final_decision,
            float(decision_confidence_threshold),
        )
        final_status += (
            f"<div>Batch ID: {html.escape(batch_id)} · "
            "Refresh Inspection History to view records and create reports.</div>"
        )
        final_image = annotated
    else:
        final_status = (
            "<section id='status-panel'>Batch complete · No supported images were found. "
            f"Batch ID: {html.escape(batch_id)}</section>"
        )
        final_image = None
    charts = _build_batch_charts(rows)
    yield (
        final_status,
        final_image,
        _filtered_rows(rows, decision_filter),
        _timing_markdown(
            len(rows),
            len(rows),
            cumulative_yolo_ms,
            cumulative_mobile_ms,
            cumulative_inference_ms,
        ),
        str(report_path),
        _batch_summary_html(rows, len(members)),
        *charts,
        rows,
    )


def remember_sample_selection(event: gr.SelectData) -> tuple[int, str]:
    index = event.index
    if isinstance(index, (tuple, list)):
        index = index[-1]
    index = int(index)
    if not 0 <= index < len(SAMPLE_IMAGES):
        raise gr.Error("Select one of the available sample images.")
    _, description = SAMPLE_IMAGES[index]
    return index, f"Selected: **{description}**"


def load_selected_sample(selected_index: int | None) -> tuple[str, str]:
    if selected_index is None or not 0 <= selected_index < len(SAMPLE_IMAGES):
        raise gr.Error("Select a sample image before loading it.")

    filename, description = SAMPLE_IMAGES[selected_index]
    image_path = SAMPLE_IMAGE_DIR / filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Sample image not found: {image_path}")

    with tempfile.NamedTemporaryFile(
        prefix="defectx_sample_", suffix=".zip", delete=False
    ) as handle:
        archive_path = Path(handle.name)
    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(image_path, arcname=filename)
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

    return (
        str(archive_path),
        f"Loaded **{description}** into the ZIP input. Click **Inspect ZIP** "
        "to run the normal inspection pipeline.",
    )


def load_sample_on_select(
    event: gr.SelectData,
) -> tuple[str, Image.Image, str, int, gr.Button]:
    index = event.index
    if isinstance(index, (tuple, list)):
        index = index[-1]
    index = int(index)
    if not 0 <= index < len(SAMPLE_IMAGES):
        raise gr.Error("Select one of the available sample images.")

    archive_path, _ = load_selected_sample(index)
    filename, description = SAMPLE_IMAGES[index]
    image_path = SAMPLE_IMAGE_DIR / filename
    with Image.open(image_path) as image:
        preview = image.convert("RGB")
    return (
        archive_path,
        preview,
        f"**Sample loaded successfully** · {description}. "
        "The one-image ZIP is already in the inspection input; inspect it below.",
        index,
        gr.Button(value="INSPECT THIS SAMPLE", interactive=True),
    )


def get_dashboard_analytics(db_path: Path | str = DATABASE_PATH):
    records = database.list_inspections(limit=10_000, db_path=db_path)
    batches = database.list_batches(limit=10_000, db_path=db_path)
    total_inspections = len(records)
    pass_count = sum(r["product_decision"] == "PASS" for r in records)
    fail_count = sum(r["product_decision"] == "FAIL" for r in records)
    review_count = sum(r["product_decision"] == "MANUAL REVIEW" for r in records)
    total_defective = sum(r["defective_objects"] for r in records)
    total_normal = sum(r["normal_objects"] for r in records)
    classified = total_defective + total_normal
    defect_rate = (
        f"{100 * total_defective / classified:.2f}%" if classified else "0.00%"
    )
    pass_rate = (
        f"{100 * pass_count / total_inspections:.2f}%" if total_inspections else "0.00%"
    )

    kpi_html = f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">TOTAL INSPECTIONS</div>
        <strong class="metric-value">{total_inspections:,}</strong>
        <div class="metric-sub">{len(batches):,} completed batch(es)</div>
      </div>
      <div class="metric-card pass">
        <div class="metric-label">PASS COUNT</div>
        <strong class="metric-value">{pass_count:,}</strong>
        <div class="metric-sub">{pass_rate} pass rate</div>
      </div>
      <div class="metric-card fail">
        <div class="metric-label">FAIL COUNT</div>
        <strong class="metric-value">{fail_count:,}</strong>
        <div class="metric-sub">{total_defective:,} defect(s) detected</div>
      </div>
      <div class="metric-card review">
        <div class="metric-label">MANUAL REVIEW</div>
        <strong class="metric-value">{review_count:,}</strong>
        <div class="metric-sub">Flagged for human QA</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">OVERALL DEFECT RATE</div>
        <strong class="metric-value">{defect_rate}</strong>
        <div class="metric-sub">Object anomaly ratio</div>
      </div>
    </div>
    """

    decision_fig = Figure(figsize=(5.5, 3.2))
    decision_ax = decision_fig.subplots()
    d_names = ("PASS", "FAIL", "MANUAL REVIEW")
    d_counts = [pass_count, fail_count, review_count]
    decision_ax.bar(d_names, d_counts, color=("#16a34a", "#dc2626", "#d97706"))
    decision_ax.set_title("Overall Inspection Decisions")
    decision_ax.set_ylabel("Inspections")

    category_counts: dict[str, int] = {}
    for r in records:
        for detail in r.get("object_details", []):
            if detail.get("status") == "defective":
                cat = detail.get("class_name", "Unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1
    cat_fig = Figure(figsize=(5.5, 3.2))
    cat_ax = cat_fig.subplots()
    categories = sorted(category_counts)
    cat_ax.bar(categories, [category_counts[c] for c in categories], color="#dc2626")
    cat_ax.set_title("Defects by Product Category")
    cat_ax.set_ylabel("Defects")
    cat_ax.tick_params(axis="x", rotation=30)
    if not categories:
        cat_ax.text(0.5, 0.5, "No confident defects in database", ha="center", va="center")

    for figure in (decision_fig, cat_fig):
        figure.patch.set_facecolor("#0b1424")
        for axis in figure.axes:
            axis.set_facecolor("#0b1424")
            axis.tick_params(colors="#b9c9e2")
            axis.xaxis.label.set_color("#91a7c7")
            axis.yaxis.label.set_color("#91a7c7")
            axis.title.set_color("#e5f1ff")
            axis.grid(axis="y", color="#283750", alpha=0.65, linewidth=0.7)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_color("#293952")
        figure.subplots_adjust(bottom=0.28)

    recent_rows = [
        [record.get(col) for col in HISTORY_COLUMNS] for record in records[:15]
    ]
    return kpi_html, decision_fig, cat_fig, recent_rows


def handle_ai_file_upload(upload) -> tuple[Image.Image | None, str, gr.Button]:
    if upload is None:
        return None, "Upload an image or ZIP file to stage for inspection.", gr.Button(interactive=False)
    if isinstance(upload, (list, tuple)) and upload:
        upload = upload[0]
    if isinstance(upload, dict):
        value = upload.get("path") or upload.get("name")
    elif isinstance(upload, (str, os.PathLike)):
        value = upload
    elif hasattr(upload, "path") and getattr(upload, "path"):
        value = upload.path
    elif hasattr(upload, "name") and getattr(upload, "name"):
        value = upload.name
    else:
        value = str(upload)
    path = Path(value)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        try:
            with Image.open(path) as img:
                return (
                    img.convert("RGB"),
                    f"Loaded **{path.name}**. Ready for inspection.",
                    gr.Button(value="RUN AI INSPECTION", interactive=True),
                )
        except Exception as e:
            return None, f"Image read error: {e}", gr.Button(interactive=False)
    elif path.suffix.lower() == ".zip":
        return (
            None,
            f"Loaded ZIP **{path.name}**. Ready for inspection.",
            gr.Button(value="RUN AI INSPECTION", interactive=True),
        )
    return None, "Unsupported file format.", gr.Button(interactive=False)


def inspect_ai_sample_or_file(
    sample_index: int | None,
    file_upload,
    yolo_confidence: float,
    yolo_iou: float,
    yolo_image_size: int,
    decision_confidence_threshold: float = 0.80,
    decision_filter: str = "ALL",
):
    target = None
    if sample_index is not None:
        try:
            s_idx = int(sample_index)
            if 0 <= s_idx < len(SAMPLE_IMAGES):
                filename, _ = SAMPLE_IMAGES[s_idx]
                target = SAMPLE_IMAGE_DIR / filename
        except (ValueError, TypeError):
            pass
    if target is None and file_upload is not None:
        target = file_upload

    if target is None:
        raise gr.Error("Please select a sample part or upload an image/ZIP archive to inspect.")

    yield from inspect_zip(
        target,
        yolo_confidence=yolo_confidence,
        yolo_iou=yolo_iou,
        yolo_image_size=yolo_image_size,
        decision_confidence_threshold=decision_confidence_threshold,
        decision_filter=decision_filter,
    )


CSS = r"""
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
  color-scheme: dark;
}
* { box-sizing: border-box; }
body, .gradio-container {
  background: var(--dx-bg) !important;
  color: var(--dx-text-main) !important;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
  max-width: 1720px !important;
  margin: 0 auto !important;
  padding: 6px 12px 32px !important;
  perspective: 1200px;
}
.main, .wrap, .contain { background: transparent !important; }

/* 3D Global Perspective on Page Layouts */
.main-view-tabs {
  perspective: 1400px;
}

/* Sidebar Navigation with 3D Depth */
.dx-sidebar {
  background: var(--dx-sidebar-bg) !important;
  border-right: 1px solid var(--dx-border) !important;
  padding: 16px 12px !important;
  box-shadow: 4px 0 24px rgba(0, 0, 0, 0.35) !important;
  transform-style: preserve-3d;
}
.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 6px 16px;
  border-bottom: 1px solid var(--dx-border);
  margin-bottom: 12px;
  transform-style: preserve-3d;
  transition: transform 0.3s ease;
}
.sidebar-brand:hover {
  transform: perspective(600px) translateZ(8px);
}
.brand-icon {
  width: 36px;
  height: 36px;
  border-radius: 9px;
  background: linear-gradient(135deg, #0284c7, #2563eb);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  box-shadow: 0 4px 16px rgba(2, 132, 199, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.3);
  flex-shrink: 0;
  transform-style: preserve-3d;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.brand-icon:hover {
  transform: rotateY(15deg) rotateX(-10deg) scale(1.05);
  box-shadow: 0 8px 24px rgba(2, 132, 199, 0.6);
}
.brand-titles {
  display: flex;
  flex-direction: column;
}
.brand-name {
  font-size: 17px;
  font-weight: 800;
  letter-spacing: 0.08em;
  color: #fff;
  line-height: 1.1;
}
.brand-name span { color: var(--dx-accent); }
.brand-sub {
  font-size: 9px;
  letter-spacing: 0.15em;
  font-family: 'DM Mono', monospace;
  color: var(--dx-text-muted);
}
.sidebar-nav {
  margin: 8px 0 16px !important;
  background: transparent !important;
  border: none !important;
}
.sidebar-nav .wrap {
  display: flex !important;
  flex-direction: column !important;
  gap: 5px !important;
}
.sidebar-nav label {
  display: flex !important;
  align-items: center !important;
  padding: 10px 14px !important;
  border-radius: 9px !important;
  background: rgba(14, 25, 45, 0.45) !important;
  border: 1px solid rgba(56, 189, 248, 0.08) !important;
  color: var(--dx-text-muted) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transform-style: preserve-3d !important;
  transition: all 0.25s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
}
.sidebar-nav label:hover {
  background: rgba(22, 38, 68, 0.75) !important;
  color: #e2e8f0 !important;
  border-color: rgba(56, 189, 248, 0.28) !important;
  transform: translateX(4px) translateZ(6px);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}
.sidebar-nav label.selected,
.sidebar-nav label:has(input:checked) {
  background: linear-gradient(135deg, rgba(2, 132, 199, 0.3), rgba(37, 99, 235, 0.3)) !important;
  border-color: rgba(56, 189, 248, 0.5) !important;
  color: var(--dx-accent) !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 16px rgba(2, 132, 199, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
  transform: translateX(4px) translateZ(8px);
}
.sidebar-nav input[type="radio"] { display: none !important; }

/* Sidebar Telemetry */
.sidebar-telemetry {
  margin-top: 18px;
  padding: 12px;
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
}
.telemetry-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 11px;
  padding: 4px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}
.telemetry-row:last-child { border-bottom: none; }
.telemetry-label { color: #94a3b8; }
.telemetry-val { font-weight: 600; color: #cbd5e1; }
.telemetry-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 6px;
}
.telemetry-dot.green { background: var(--dx-pass); box-shadow: 0 0 8px var(--dx-pass); }
.telemetry-dot.cyan { background: var(--dx-accent); box-shadow: 0 0 8px var(--dx-accent); }
.telemetry-dot.blue { background: #3b82f6; box-shadow: 0 0 8px #3b82f6; }

/* Main Tabs Navigation */
.main-view-tabs > .tab-nav {
  display: flex !important;
  gap: 6px !important;
  padding: 6px !important;
  background: rgba(11, 20, 36, 0.75) !important;
  border: 1px solid var(--dx-border) !important;
  border-radius: 10px !important;
  margin-bottom: 16px !important;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
}
.main-view-tabs > .tab-nav button {
  padding: 8px 16px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  border-radius: 7px !important;
  color: var(--dx-text-muted) !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  transition: all 0.2s ease !important;
}
.main-view-tabs > .tab-nav button:hover {
  color: #fff !important;
  background: rgba(30, 41, 59, 0.6) !important;
  transform: translateY(-1px);
}
.main-view-tabs > .tab-nav button.selected {
  background: linear-gradient(135deg, rgba(2, 132, 199, 0.25), rgba(37, 99, 235, 0.25)) !important;
  color: var(--dx-accent) !important;
  border: 1px solid var(--dx-border-active) !important;
  box-shadow: 0 2px 10px rgba(2, 132, 199, 0.25) !important;
}

/* Card & Panel Styling with 3D Depth */
.app-card, .control-column, .viewport-column, .specs-card {
  background: var(--dx-panel-bg) !important;
  border: 1px solid var(--dx-border) !important;
  border-radius: 12px !important;
  padding: 16px !important;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.28), inset 0 1px 1px rgba(255, 255, 255, 0.08) !important;
  transform-style: preserve-3d;
  transition: transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.3s ease, border-color 0.25s ease;
}
.panel-kicker {
  font-size: 11px;
  font-family: 'DM Mono', monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--dx-accent);
  margin-bottom: 3px;
  transform: translateZ(6px);
}
.panel-title {
  font-size: 16px;
  font-weight: 700;
  color: #fff;
  margin-top: 0;
  margin-bottom: 12px;
  transform: translateZ(10px);
}

/* 3D Interactive KPI Cards */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin-bottom: 18px;
  perspective: 1000px;
}
.metric-card {
  background: rgba(11, 20, 36, 0.88);
  border: 1px solid var(--dx-border);
  border-radius: 11px;
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  transform-style: preserve-3d;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.08);
  transition: transform 0.28s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.28s ease, border-color 0.2s ease;
  cursor: default;
}
.metric-card:hover {
  transform: perspective(800px) translateY(-5px) translateZ(14px) rotateX(2deg);
  border-color: rgba(56, 189, 248, 0.45) !important;
  box-shadow: 0 16px 36px -6px rgba(0, 0, 0, 0.55), 0 0 24px rgba(56, 189, 248, 0.22);
}
.metric-card .metric-label {
  font-size: 10.5px;
  letter-spacing: 0.08em;
  font-family: 'DM Mono', monospace;
  color: #94a3b8;
  text-transform: uppercase;
  transform: translateZ(8px);
}
.metric-card .metric-value {
  font-size: 28px;
  font-weight: 700;
  color: #f1f5f9;
  line-height: 1.1;
  margin: 4px 0;
  transform: translateZ(22px);
  text-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}
.metric-card .metric-sub {
  font-size: 11.5px;
  color: #64748b;
  transform: translateZ(6px);
}
.metric-card.pass { border-left: 3px solid var(--dx-pass); }
.metric-card.pass .metric-value { color: #34d399; text-shadow: 0 0 16px rgba(16, 185, 129, 0.3); }
.metric-card.fail { border-left: 3px solid var(--dx-fail); }
.metric-card.fail .metric-value { color: #f87171; text-shadow: 0 0 16px rgba(239, 68, 68, 0.3); }
.metric-card.review { border-left: 3px solid var(--dx-review); }
.metric-card.review .metric-value { color: #fbbf24; text-shadow: 0 0 16px rgba(245, 158, 11, 0.3); }

/* 3D Holographic Spatial Sensing Node on Dashboard */
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
  transform-style: preserve-3d;
  position: relative;
  overflow: hidden;
  transition: transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1), border-color 0.3s ease;
}
.hologram-card:hover {
  border-color: rgba(56, 189, 248, 0.45);
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.55), 0 0 30px rgba(2, 132, 199, 0.25);
}
.hologram-scene {
  position: relative;
  width: 240px;
  height: 180px;
  margin: 0 auto;
  perspective: 900px;
  transform-style: preserve-3d;
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

/* 3D Wireframe Floating Inspection Cube */
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

/* 3D Dual Viewport Row & Image Frames */
.workspace-row { align-items: stretch !important; gap: 14px !important; }
.dual-viewport-row {
  perspective: 1200px;
  gap: 14px !important;
}
.viewport-box {
  position: relative;
  transform-style: preserve-3d;
  border-radius: 12px !important;
  background: linear-gradient(135deg, rgba(16, 29, 52, 0.7), rgba(8, 15, 28, 0.9)) !important;
  border: 1px solid rgba(56, 189, 248, 0.22) !important;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.12) !important;
  transition: transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.35s ease, border-color 0.25s ease !important;
  overflow: hidden;
}
.viewport-box:hover {
  transform: perspective(1000px) translateY(-4px) translateZ(14px) rotateX(1deg);
  border-color: rgba(56, 189, 248, 0.45) !important;
  box-shadow: 0 20px 45px rgba(0, 0, 0, 0.5), 0 0 30px rgba(2, 132, 199, 0.28), inset 0 1px 2px rgba(255, 255, 255, 0.2) !important;
}
/* 3D Laser Scanning Beam Effect on Viewports */
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

/* Sample Gallery */
#sample-gallery {
  border: 1px solid rgba(56, 189, 248, 0.18);
  border-radius: 10px;
  padding: 6px;
  background: rgba(8, 14, 26, 0.7);
  perspective: 800px;
}
#sample-gallery button.thumbnail-item {
  border: 1px solid rgba(100, 140, 200, 0.18);
  border-radius: 8px;
  background: #0d1728;
  transform-style: preserve-3d;
  transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}
#sample-gallery button.thumbnail-item:hover {
  transform: perspective(600px) translateY(-3px) translateZ(10px) rotateX(2deg);
  border-color: var(--dx-accent);
  box-shadow: 0 6px 18px var(--dx-accent-glow);
}
#sample-gallery button.thumbnail-item.selected {
  border-color: var(--dx-accent);
  box-shadow: 0 0 0 2px var(--dx-accent-glow);
  transform: perspective(600px) translateZ(12px);
}

/* 3D Tactile Form Controls & Buttons */
.gradio-container input, .gradio-container textarea, .gradio-container select {
  background: #0b1526 !important;
  color: #f1f5f9 !important;
  border: 1px solid rgba(100, 140, 200, 0.2) !important;
  border-radius: 8px !important;
  transition: all 0.2s ease !important;
}
.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container select:focus {
  border-color: var(--dx-accent) !important;
  box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.22) !important;
}
.gradio-container button {
  border-radius: 8px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  transform-style: preserve-3d;
  transition: all 0.2s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
}
.gradio-container button.primary, .gradio-container button[variant="primary"] {
  background: linear-gradient(135deg, #0284c7, #2563eb 60%, #4f46e5) !important;
  border: 1px solid rgba(56, 189, 248, 0.45) !important;
  color: #fff !important;
  box-shadow: 0 4px 0 #1d4ed8, 0 8px 18px rgba(0, 0, 0, 0.35) !important;
}
.gradio-container button.primary:hover {
  transform: translateY(-2px) translateZ(6px);
  box-shadow: 0 6px 0 #1d4ed8, 0 12px 24px rgba(2, 132, 199, 0.4) !important;
}
.gradio-container button.primary:active {
  transform: translateY(3px) translateZ(-2px) !important;
  box-shadow: 0 1px 0 #1d4ed8, 0 3px 8px rgba(0, 0, 0, 0.4) !important;
}
.gradio-container button.secondary, .gradio-container button[variant="secondary"] {
  background: rgba(20, 35, 60, 0.7) !important;
  border: 1px solid var(--dx-border) !important;
  color: #cbd5e1 !important;
  box-shadow: 0 3px 0 rgba(10, 18, 32, 0.8), 0 6px 14px rgba(0, 0, 0, 0.25) !important;
}
.gradio-container button.secondary:hover {
  background: rgba(30, 50, 85, 0.85) !important;
  color: #fff !important;
  transform: translateY(-1px) translateZ(4px);
}
.gradio-container button.secondary:active {
  transform: translateY(2px) translateZ(-1px) !important;
  box-shadow: 0 1px 0 rgba(10, 18, 32, 0.8), 0 2px 4px rgba(0, 0, 0, 0.25) !important;
}

/* Status & Timing Panels with 3D Pop */
#status-panel { border-radius: 10px; overflow: hidden; }
.decision-result {
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.12);
  transform-style: preserve-3d;
  transition: transform 0.3s ease;
}
.decision-result:hover {
  transform: perspective(800px) translateZ(10px);
}
#timing-panel {
  border: 1px solid rgba(56, 189, 248, 0.15);
  border-radius: 8px;
  background: rgba(9, 18, 33, 0.6);
  padding: 8px 12px;
  font-size: 12.5px;
}
#timing-panel p { margin: 0; line-height: 1.5; }

/* Tables & Plots */
.gradio-container .plot-container {
  border: 1px solid var(--dx-border);
  border-radius: 10px;
  background: #091220;
  padding: 6px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.22);
}
.gradio-container .dataframe, .gradio-container table {
  border: 1px solid var(--dx-border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}
.gradio-container th {
  background: #101c30 !important;
  color: #cbd5e1 !important;
  font-size: 11.5px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
}
.gradio-container td {
  background: rgba(10, 18, 32, 0.8) !important;
  color: #e2e8f0 !important;
  border-color: rgba(100, 140, 200, 0.1) !important;
}

/* History Details */
#history-details {
  border: 1px solid var(--dx-border);
  border-radius: 10px;
  padding: 14px;
  background: rgba(10, 18, 32, 0.8);
}
#history-details h3 { color: var(--dx-accent) !important; margin-top: 0; }

/* 3D Isometric Floating Workflow Stations in About Page */
.workflow-flow {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 22px 14px;
  background: rgba(9, 17, 31, 0.7);
  border: 1px solid var(--dx-border);
  border-radius: 10px;
  margin-top: 12px;
  overflow-x: auto;
  perspective: 1000px;
}
.flow-step {
  text-align: center;
  min-width: 115px;
  padding: 12px 10px;
  border-radius: 8px;
  background: rgba(14, 25, 45, 0.65);
  border: 1px solid rgba(56, 189, 248, 0.15);
  transform-style: preserve-3d;
  transform: perspective(600px) rotateX(10deg);
  transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1);
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35);
}
.flow-step:hover {
  transform: perspective(600px) rotateX(0deg) translateY(-8px) translateZ(18px) scale(1.05);
  box-shadow: 0 16px 30px rgba(2, 132, 199, 0.3), 0 0 18px rgba(56, 189, 248, 0.35);
  border-color: rgba(56, 189, 248, 0.5);
}
.flow-step span {
  display: block;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--dx-accent);
  margin-bottom: 2px;
  transform: translateZ(8px);
}
.flow-step b {
  display: block;
  color: #f1f5f9;
  font-size: 13px;
  transform: translateZ(12px);
}
.flow-step p {
  color: var(--dx-text-muted);
  font-size: 11px;
  margin: 2px 0 0;
  transform: translateZ(6px);
}
.flow-arrow {
  color: var(--dx-accent);
  font-size: 18px;
  font-weight: 700;
  text-shadow: 0 0 8px rgba(56, 189, 248, 0.5);
  animation: arrowPulse 2s ease-in-out infinite;
}
@keyframes arrowPulse {
  0%, 100% { transform: scale(1); opacity: 0.7; }
  50% { transform: scale(1.2); opacity: 1; }
}
.specs-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 14px;
}
.specs-item {
  background: rgba(10, 18, 32, 0.7);
  border: 1px solid rgba(100, 140, 200, 0.14);
  border-radius: 8px;
  padding: 14px;
  transform-style: preserve-3d;
  transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}
.specs-item:hover {
  transform: perspective(800px) translateY(-3px) translateZ(10px);
  border-color: rgba(56, 189, 248, 0.4);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
}
.specs-item h4 {
  color: var(--dx-accent);
  margin-top: 0;
  margin-bottom: 6px;
  font-size: 13.5px;
}
.hint-text {
  font-size: 11.5px !important;
  color: var(--dx-text-muted) !important;
  margin-top: -4px !important;
  margin-bottom: 6px !important;
}
@media (max-width: 900px) {
  body, .gradio-container { padding: 4px 6px 20px !important; }
  .hologram-card { grid-template-columns: 1fr; }
  .workspace-row { flex-direction: column !important; }
  .dual-viewport-row { flex-direction: column !important; }
  .workflow-flow { flex-direction: column; align-items: flex-start; }
  .flow-arrow { display: none; }
}
"""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="DefectX | Industrial Quality Inspection Software") as demo:
        with gr.Sidebar(open=True, elem_classes=["dx-sidebar"]):
            gr.HTML(
                '<div class="sidebar-brand">'
                '<div class="brand-icon">'
                '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>'
                '</svg></div>'
                '<div class="brand-titles">'
                '<div class="brand-name"><span>DEFECT</span>X</div>'
                '<div class="brand-sub">QUALITY INTELLIGENCE</div>'
                '</div></div>'
            )
            nav = gr.Radio(
                choices=[
                    "📊 Dashboard",
                    "🔬 AI Inspection",
                    "📦 Batch Inspection",
                    "🗃️ Inspection History",
                    "📄 Reports",
                    "ℹ️ About the System",
                ],
                value="📊 Dashboard",
                label="MAIN NAVIGATION",
                elem_classes=["sidebar-nav"],
            )
            gr.HTML(
                '<div class="sidebar-telemetry">'
                '<div class="telemetry-header">SYSTEM ARCHITECTURE</div>'
                '<div class="telemetry-row"><span class="telemetry-dot green"></span>'
                '<span class="telemetry-label">YOLO11 Segmentation</span><span class="telemetry-val">Online</span></div>'
                '<div class="telemetry-row"><span class="telemetry-dot green"></span>'
                '<span class="telemetry-label">MobileNetV3 Classifier</span><span class="telemetry-val">Online</span></div>'
                '<div class="telemetry-row"><span class="telemetry-dot cyan"></span>'
                '<span class="telemetry-label">SQLite Audit Trail</span><span class="telemetry-val">WAL Active</span></div>'
                '<div class="telemetry-row"><span class="telemetry-dot blue"></span>'
                '<span class="telemetry-label">Dataset Benchmark</span><span class="telemetry-val">VisA Industrial</span></div>'
                '</div>'
            )

        with gr.Tabs(elem_classes=["main-view-tabs"]) as main_tabs:
            # -------------------------------------------------------------
            # PAGE 1: Dashboard Overview
            # -------------------------------------------------------------
            with gr.Tab("📊 Dashboard", id="tab_dashboard"):
                with gr.Row():
                    gr.HTML(
                        '<div class="panel-kicker">EXECUTIVE QUALITY INTELLIGENCE</div>'
                        '<div class="panel-title">Quality Assurance Dashboard Overview</div>'
                    )
                    dashboard_refresh_btn = gr.Button(
                        "🔄 Refresh Metrics", variant="secondary", size="sm"
                    )
                gr.HTML(
                    """<div class="hologram-card">
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
                    </div>"""
                )
                dashboard_kpi_html = gr.HTML(
                    """<div class="metric-grid">
                      <div class="metric-card"><div class="metric-label">TOTAL INSPECTIONS</div><strong class="metric-value">0</strong></div>
                      <div class="metric-card pass"><div class="metric-label">PASS COUNT</div><strong class="metric-value">0</strong></div>
                      <div class="metric-card fail"><div class="metric-label">FAIL COUNT</div><strong class="metric-value">0</strong></div>
                      <div class="metric-card review"><div class="metric-label">MANUAL REVIEW</div><strong class="metric-value">0</strong></div>
                      <div class="metric-card"><div class="metric-label">DEFECT RATE</div><strong class="metric-value">0.00%</strong></div>
                    </div>"""
                )
                with gr.Row():
                    dashboard_decision_chart = gr.Plot(label="Overall Quality Decisions")
                    dashboard_category_chart = gr.Plot(label="Defects by Product Category")
                gr.HTML(
                    '<div class="panel-kicker" style="margin-top:16px;">RECENT AUDIT TRAIL</div>'
                    '<div class="panel-title">Recent Inspection Records</div>'
                )
                dashboard_recent_table = gr.Dataframe(
                    headers=HISTORY_COLUMNS,
                    datatype=[
                        "str", "str", "str", "str", "number", "number", "number",
                        "number", "str", "number",
                    ],
                    label="Recent Inspection Records",
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    quick_single_btn = gr.Button("🔬 New AI Inspection →", variant="primary")
                    quick_batch_btn = gr.Button("📦 New Batch Inspection →", variant="secondary")

            # -------------------------------------------------------------
            # PAGE 2: AI Inspection Workspace
            # -------------------------------------------------------------
            with gr.Tab("🔬 AI Inspection", id="tab_ai_inspect"):
                gr.HTML(
                    '<div class="panel-kicker">REAL-TIME ANOMALY DETECTION</div>'
                    '<div class="panel-title">AI Quality Inspection Workspace</div>'
                )
                gr.Markdown(
                    "**Inspection Workflow:** 1. Select a VisA sample part or upload an image. "
                    "2. Set the operational confidence threshold (default 0.80). "
                    "3. Click **Run AI Inspection** to execute YOLO11 segmentation and MobileNetV3 defect classification.",
                    elem_classes=["hint-text"],
                )
                with gr.Row(elem_classes=["workspace-row"]):
                    # Left Column: Source Ingestion & Parameters
                    with gr.Column(scale=5, elem_classes=["control-column"]):
                        gr.HTML(
                            '<div class="panel-kicker">SOURCE INGESTION</div>'
                            '<div class="panel-title">Inspection Input</div>'
                        )
                        with gr.Tabs():
                            with gr.Tab("🖼️ Sample Parts"):
                                gr.Markdown(
                                    "Click any preloaded VisA part to stage for inspection:",
                                    elem_classes=["hint-text"],
                                )
                                sample_gallery = gr.Gallery(
                                    value=[
                                        (str(SAMPLE_IMAGE_DIR / filename), description)
                                        for filename, description in SAMPLE_IMAGES
                                    ],
                                    label="VisA Dataset Product Samples",
                                    columns=3,
                                    height=230,
                                    allow_preview=False,
                                    object_fit="contain",
                                    interactive=False,
                                    elem_id="sample-gallery",
                                )
                                selected_sample = gr.State(value=None)
                                with gr.Row():
                                    selection_status = gr.Markdown(
                                        "Click a sample part above.",
                                        elem_id="sample-status",
                                    )
                                    sample_inspect_button = gr.Button(
                                        "INSPECT THIS SAMPLE",
                                        variant="primary",
                                        interactive=False,
                                        elem_id="sample-inspect-button",
                                    )
                            with gr.Tab("📁 Upload Image / ZIP"):
                                gr.Markdown(
                                    "Upload a single product image or ZIP archive:",
                                    elem_classes=["hint-text"],
                                )
                                ai_file = gr.File(
                                    label="Upload Product Image or ZIP Archive",
                                    file_types=[
                                        ".zip", ".jpg", ".jpeg", ".png",
                                        ".bmp", ".webp", ".tif", ".tiff",
                                    ],
                                    type="filepath",
                                )
                                gr.Markdown(
                                    "**Supported formats:** JPG, JPEG, PNG, BMP, WEBP, TIF, TIFF, ZIP  \n"
                                    "**Limits:** 100 MiB per image · 4 GiB max archive",
                                    elem_classes=["hint-text"],
                                )

                        with gr.Accordion("⚙️ Inspection Thresholds & Parameters", open=True):
                            confidence = gr.Slider(
                                0.05, 0.95, value=0.25, step=0.05, label="YOLO segmentation confidence"
                            )
                            iou = gr.Slider(
                                0.10, 0.90, value=0.70, step=0.05, label="YOLO IoU threshold"
                            )
                            decision_confidence = gr.Slider(
                                0.50,
                                1.00,
                                value=0.80,
                                step=0.01,
                                label="Operational decision confidence",
                            )
                            gr.Markdown(
                                "Operational threshold for routing uncertain cases to review. "
                                "Trained MobileNet checkpoint threshold remains unchanged.",
                                elem_classes=["hint-text"],
                            )
                            image_size = gr.Dropdown(
                                choices=[416, 512, 640, 768, 960, 1024],
                                value=640,
                                label="YOLO inference size",
                            )

                        ai_run_button = gr.Button(
                            "⚡ RUN AI INSPECTION",
                            variant="primary",
                            elem_id="run-zip-inspection",
                        )

                    # Right Column: Side-by-Side Viewports & Telemetry
                    with gr.Column(scale=7, elem_classes=["viewport-column"]):
                        ai_status = gr.HTML(
                            "<div id='status-panel'>Models loaded · Ready for inspection. "
                            "Select a sample part or upload an image to begin.</div>"
                        )
                        with gr.Row(elem_classes=["dual-viewport-row"]):
                            ai_original_image = gr.Image(
                                label="Original Sensor Input Image",
                                type="pil",
                                height=380,
                                elem_classes=["viewport-box"],
                            )
                            ai_annotated_image = gr.Image(
                                label="AI Defect Segmentation Overlay",
                                type="pil",
                                height=380,
                                elem_classes=["viewport-box"],
                            )
                        ai_timing = gr.Markdown(
                            _timing_markdown(0, 0, 0.0, 0.0, 0.0),
                            elem_id="timing-panel",
                        )

            # -------------------------------------------------------------
            # PAGE 3: Batch Inspection
            # -------------------------------------------------------------
            with gr.Tab("📦 Batch Inspection", id="tab_batch_inspect"):
                gr.HTML(
                    '<div class="panel-kicker">HIGH-THROUGHPUT QUALITY CONTROL</div>'
                    '<div class="panel-title">Batch Inspection & Quality Analytics</div>'
                )
                with gr.Row(elem_classes=["workspace-row"]):
                    with gr.Column(scale=5, elem_classes=["control-column"]):
                        gr.HTML(
                            '<div class="panel-kicker">BATCH INGESTION</div>'
                            '<div class="panel-title">Upload ZIP Archive</div>'
                        )
                        batch_zip_file = gr.File(
                            label="Drop ZIP archive here",
                            file_types=[".zip"],
                            type="filepath",
                        )
                        gr.Markdown(
                            "**Supported images:** JPG, JPEG, PNG, BMP, WEBP, TIF, TIFF  \n"
                            "**Limits:** 10,000 images · 100 MiB / image · 4 GiB total",
                            elem_classes=["hint-text"],
                        )
                        with gr.Accordion("⚙️ Batch Parameters", open=False):
                            batch_confidence = gr.Slider(
                                0.05, 0.95, value=0.25, step=0.05, label="YOLO confidence"
                            )
                            batch_iou = gr.Slider(
                                0.10, 0.90, value=0.70, step=0.05, label="YOLO IoU"
                            )
                            batch_decision_confidence = gr.Slider(
                                0.50, 1.00, value=0.80, step=0.01, label="Decision confidence threshold"
                            )
                            batch_image_size = gr.Dropdown(
                                choices=[416, 512, 640, 768, 960, 1024],
                                value=640,
                                label="YOLO inference size",
                            )
                        batch_run_button = gr.Button(
                            "⚡ RUN BATCH INSPECTION",
                            variant="primary",
                        )
                    with gr.Column(scale=7, elem_classes=["viewport-column"]):
                        batch_status = gr.HTML(
                            "<div id='status-panel'>Upload a ZIP batch to begin high-throughput quality analysis.</div>"
                        )
                        batch_current_image = gr.Image(
                            label="Active Processing Viewport",
                            type="pil",
                            height=320,
                        )
                        batch_timing = gr.Markdown(
                            _timing_markdown(0, 0, 0.0, 0.0, 0.0),
                            elem_id="timing-panel",
                        )

                batch_summary = gr.HTML(
                    _batch_summary_html([], 0), label="Batch summary"
                )
                with gr.Row():
                    decision_chart = gr.Plot(label="PASS / FAIL / MANUAL REVIEW Decisions")
                    object_chart = gr.Plot(label="Normal / Defective / Uncertain Objects")
                with gr.Row():
                    category_chart = gr.Plot(label="Defects by Product Category")
                    inference_time_chart = gr.Plot(label="Per-Image Latency Distribution")
                with gr.Row():
                    decision_filter = gr.Radio(
                        choices=["ALL", "PASS", "FAIL", "MANUAL REVIEW"],
                        value="ALL",
                        label="Filter image-level results",
                    )
                    report_file = gr.File(
                        label="Download Complete Batch CSV Report",
                        interactive=False,
                    )
                batch_rows_state = gr.State(value=[])
                results = gr.Dataframe(
                    headers=RESULT_COLUMNS,
                    datatype=[
                        "str", "str", "number", "number", "str", "str", "str", "str",
                        "str", "str", "str", "str", "number", "number", "number", "str",
                        "str", "number",
                    ],
                    label="Image-Wise Inspection Results",
                    interactive=False,
                    wrap=True,
                    elem_classes=["results-table"],
                )

            # -------------------------------------------------------------
            # PAGE 4: Inspection History
            # -------------------------------------------------------------
            with gr.Tab("🗃️ Inspection History", id="tab_history"):
                gr.HTML(
                    '<div class="panel-kicker">COMPLIANCE & TRACEABILITY</div>'
                    '<div class="panel-title">Inspection Audit Trail Database</div>'
                )
                with gr.Row():
                    history_search = gr.Textbox(
                        label="Search by inspection ID or image filename",
                        placeholder="e.g. candle_Normal_0085.JPG",
                        scale=3,
                    )
                    history_decision = gr.Dropdown(
                        choices=["ALL", "PASS", "FAIL", "MANUAL REVIEW"],
                        value="ALL",
                        label="Decision Filter",
                        scale=1,
                    )
                    history_refresh = gr.Button(
                        "SEARCH HISTORY",
                        variant="primary",
                        scale=1,
                    )
                history_summary = gr.HTML("History records load when the page opens.")
                history_table = gr.Dataframe(
                    headers=HISTORY_COLUMNS,
                    datatype=[
                        "str", "str", "str", "str", "number", "number", "number",
                        "number", "str", "number",
                    ],
                    label="Saved Inspection Records",
                    interactive=False,
                    wrap=True,
                )

                gr.HTML(
                    '<div class="panel-kicker" style="margin-top:16px;">RECORD VERIFICATION</div>'
                    '<div class="panel-title">Inspect Selected Record & Evidence</div>'
                )
                with gr.Row():
                    history_inspection_id = gr.Dropdown(
                        choices=[],
                        label="Select an inspection record",
                        interactive=False,
                        scale=3,
                    )
                    history_view_button = gr.Button("VIEW RECORD DETAILS", scale=1)
                    individual_pdf_button = gr.Button(
                        "DOWNLOAD RECORD PDF", variant="primary", scale=1
                    )
                history_details = gr.HTML(elem_id="history-details")
                with gr.Row(elem_classes=["dual-viewport-row"]):
                    original_record_image = gr.Image(
                        label="Original Raw Sensor Capture",
                        type="pil",
                        height=320,
                        elem_classes=["viewport-box"],
                    )
                    annotated_record_image = gr.Image(
                        label="AI Annotated Defect Segmentation",
                        type="pil",
                        height=320,
                        elem_classes=["viewport-box"],
                    )
                individual_pdf_file = gr.File(
                    label="Individual Inspection PDF Report",
                    interactive=False,
                )

                gr.HTML(
                    '<div class="panel-kicker" style="margin-top:16px;">BATCH AUDIT ARCHIVE</div>'
                    '<div class="panel-title">Consolidated Batch Quality Report</div>'
                )
                with gr.Row():
                    history_batch_id = gr.Dropdown(
                        choices=[],
                        label="Select a completed batch",
                        interactive=False,
                        scale=3,
                    )
                    batch_pdf_button = gr.Button(
                        "DOWNLOAD BATCH PDF", variant="primary", scale=1
                    )
                batch_pdf_file = gr.File(
                    label="Consolidated Batch Quality PDF",
                    interactive=False,
                )

            # -------------------------------------------------------------
            # PAGE 5: Reports Center
            # -------------------------------------------------------------
            with gr.Tab("📄 Reports", id="tab_reports"):
                gr.HTML(
                    '<div class="panel-kicker">QUALITY COMPLIANCE & EXPORTS</div>'
                    '<div class="panel-title">Inspection Reports Center</div>'
                )
                with gr.Row():
                    reports_refresh_btn = gr.Button(
                        "🔄 Refresh Available Records & Batches", variant="secondary"
                    )
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["control-column"]):
                        gr.HTML(
                            '<div class="panel-kicker">INDIVIDUAL CERTIFICATION</div>'
                            '<div class="panel-title">Single Inspection PDF Certificate</div>'
                        )
                        reports_inspection_id = gr.Dropdown(
                            choices=[],
                            label="Select Inspection Record (by ID or filename)",
                            interactive=False,
                        )
                        reports_individual_pdf_btn = gr.Button(
                            "📄 Generate & Download Inspection PDF", variant="primary"
                        )
                        reports_individual_pdf_file = gr.File(
                            label="Inspection Certificate PDF", interactive=False
                        )
                    with gr.Column(scale=1, elem_classes=["control-column"]):
                        gr.HTML(
                            '<div class="panel-kicker">CONSOLIDATED BATCH AUDIT</div>'
                            '<div class="panel-title">Batch Inspection Quality Report</div>'
                        )
                        reports_batch_id = gr.Dropdown(
                            choices=[],
                            label="Select Completed Batch Record",
                            interactive=False,
                        )
                        reports_batch_pdf_btn = gr.Button(
                            "📑 Generate & Download Batch PDF", variant="primary"
                        )
                        reports_batch_pdf_file = gr.File(
                            label="Consolidated Batch PDF Report", interactive=False
                        )
                gr.HTML(
                    '<div class="specs-card" style="margin-top:16px;">'
                    '<div class="panel-kicker">AUDIT COMPLIANCE STANDARDS</div>'
                    '<div class="panel-title">ISO 9001 Inspection Documentation</div>'
                    '<p style="color:#94a3b8;font-size:13px;line-height:1.6">'
                    'DefectX generates tamper-evident, uncompressed vector PDF reports designed for ISO 9001 and IATF 16949 '
                    'quality compliance audits. Every report documents the exact timestamp, product category, individual object coordinates, '
                    'classification confidences, and model decision thresholds.'
                    '</p></div>'
                )

            # -------------------------------------------------------------
            # PAGE 6: About the System
            # -------------------------------------------------------------
            with gr.Tab("ℹ️ About the System", id="tab_about"):
                gr.HTML(
                    '<div class="specs-card">'
                    '<div class="panel-kicker">SYSTEM ARCHITECTURE</div>'
                    '<div class="panel-title">DefectX Intelligent Inspection Workflow</div>'
                    '<div class="workflow-flow">'
                    '<div class="flow-step"><span>01</span><b>Raw Capture</b><p>Image Acquisition</p></div>'
                    '<div class="flow-arrow">→</div>'
                    '<div class="flow-step"><span>02</span><b>YOLO11</b><p>Instance Segmentation</p></div>'
                    '<div class="flow-arrow">→</div>'
                    '<div class="flow-step"><span>03</span><b>MobileNetV3</b><p>Anomaly Classification</p></div>'
                    '<div class="flow-arrow">→</div>'
                    '<div class="flow-step"><span>04</span><b>Decision Engine</b><p>Threshold Routing</p></div>'
                    '<div class="flow-arrow">→</div>'
                    '<div class="flow-step"><span>05</span><b>SQLite Audit</b><p>Persistent Trail</p></div>'
                    '<div class="flow-arrow">→</div>'
                    '<div class="flow-step"><span>06</span><b>PDF Report</b><p>Quality Certification</p></div>'
                    '</div>'
                    '<div class="specs-grid" style="margin-top:20px;">'
                    '<div class="specs-item">'
                    '<h4>Dataset & Model Grounding</h4>'
                    '<p style="color:#94a3b8;font-size:13px;line-height:1.6">'
                    'The current vision models are trained and benchmarked on the <b>VisA (Visual Anomaly)</b> industrial dataset, '
                    'covering product categories such as candles, capsules, and printed circuit boards (PCBs). '
                    '<b>Important validation notice:</b> This system is designed for automated defect inspection research, evaluation, '
                    'and prototyping. The current models are grounded strictly on the VisA dataset and do not represent '
                    'automotive-specific certification or production plant deployment validation without custom domain calibration.'
                    '</p></div>'
                    '<div class="specs-item">'
                    '<h4>Operational Decision Policy</h4>'
                    '<p style="color:#94a3b8;font-size:13px;line-height:1.6">'
                    '<b>PASS:</b> All detected objects are classified as normal with confidence ≥ operational threshold.<br>'
                    '<b>FAIL:</b> One or more objects are classified as defective with confidence ≥ operational threshold.<br>'
                    '<b>MANUAL REVIEW:</b> Any object classification confidence is below threshold, zero objects detected, '
                    'or an image decode error occurred. Prevents false passes on uncertain edge cases.'
                    '</p></div>'
                    '<div class="specs-item">'
                    '<h4>Neural Network Architecture</h4>'
                    '<p style="color:#94a3b8;font-size:13px;line-height:1.6">'
                    '<b>Stage 1:</b> YOLO11 retina segmentation head for precise polygonal boundary extraction.<br>'
                    '<b>Stage 2:</b> MobileNetV3-Large feature backbone with 128-dimensional class embedding and FiLM '
                    '(Feature-wise Linear Modulation) conditioning for context-aware anomaly scoring.'
                    '</p></div>'
                    '<div class="specs-item">'
                    '<h4>Audit & Persistence Layer</h4>'
                    '<p style="color:#94a3b8;font-size:13px;line-height:1.6">'
                    'Structured SQLite database in WAL (Write-Ahead Logging) mode with foreign key enforcement. '
                    'Stores raw images, annotated overlays, JSON metadata, and millisecond latency telemetry.'
                    '</p></div>'
                    '</div></div>'
                )

        # -----------------------------------------------------------------
        # Sidebar & Tab Synchronization
        # -----------------------------------------------------------------
        def on_nav_change(choice: str):
            tab_map = {
                "📊 Dashboard": "tab_dashboard",
                "🔬 AI Inspection": "tab_ai_inspect",
                "📦 Batch Inspection": "tab_batch_inspect",
                "🗃️ Inspection History": "tab_history",
                "📄 Reports": "tab_reports",
                "ℹ️ About the System": "tab_about",
            }
            return gr.update(selected=tab_map.get(choice, "tab_dashboard"))

        nav.change(fn=on_nav_change, inputs=[nav], outputs=[main_tabs])

        quick_single_btn.click(
            fn=lambda: (gr.update(value="🔬 AI Inspection"), gr.update(selected="tab_ai_inspect")),
            outputs=[nav, main_tabs],
        )
        quick_batch_btn.click(
            fn=lambda: (gr.update(value="📦 Batch Inspection"), gr.update(selected="tab_batch_inspect")),
            outputs=[nav, main_tabs],
        )

        dashboard_refresh_btn.click(
            fn=get_dashboard_analytics,
            inputs=None,
            outputs=[
                dashboard_kpi_html,
                dashboard_decision_chart,
                dashboard_category_chart,
                dashboard_recent_table,
            ],
        )

        # -----------------------------------------------------------------
        # AI Inspection Event Bindings
        # -----------------------------------------------------------------
        def on_ai_file_upload(upload_file):
            preview, msg, btn = handle_ai_file_upload(upload_file)
            return preview, msg, btn, None

        ai_file.upload(
            fn=on_ai_file_upload,
            inputs=[ai_file],
            outputs=[
                ai_original_image,
                selection_status,
                sample_inspect_button,
                selected_sample,
            ],
        )

        ai_inspection_inputs = [
            selected_sample,
            ai_file,
            confidence,
            iou,
            image_size,
            decision_confidence,
            decision_filter,
        ]
        ai_inspection_outputs = [
            ai_status,
            ai_annotated_image,
            results,
            ai_timing,
            report_file,
            batch_summary,
            decision_chart,
            object_chart,
            category_chart,
            inference_time_chart,
            batch_rows_state,
        ]

        sample_gallery.select(
            fn=load_sample_on_select,
            inputs=None,
            outputs=[
                ai_file,
                ai_original_image,
                selection_status,
                selected_sample,
                sample_inspect_button,
            ],
        ).then(
            fn=inspect_ai_sample_or_file,
            inputs=ai_inspection_inputs,
            outputs=ai_inspection_outputs,
            show_progress="minimal",
        )

        ai_run_button.click(
            fn=inspect_ai_sample_or_file,
            inputs=ai_inspection_inputs,
            outputs=ai_inspection_outputs,
            show_progress="minimal",
        )
        sample_inspect_button.click(
            fn=inspect_ai_sample_or_file,
            inputs=ai_inspection_inputs,
            outputs=ai_inspection_outputs,
            show_progress="minimal",
        )

        # -----------------------------------------------------------------
        # Batch Inspection Event Bindings
        # -----------------------------------------------------------------
        batch_run_button.click(
            fn=inspect_zip,
            inputs=[
                batch_zip_file,
                batch_confidence,
                batch_iou,
                batch_image_size,
                batch_decision_confidence,
                decision_filter,
            ],
            outputs=[
                batch_status,
                batch_current_image,
                results,
                batch_timing,
                report_file,
                batch_summary,
                decision_chart,
                object_chart,
                category_chart,
                inference_time_chart,
                batch_rows_state,
            ],
            show_progress="minimal",
        )
        decision_filter.change(
            fn=filter_batch_results,
            inputs=[decision_filter, batch_rows_state],
            outputs=[results],
        )

        # -----------------------------------------------------------------
        # History Event Bindings
        # -----------------------------------------------------------------
        history_refresh.click(
            fn=refresh_inspection_history,
            inputs=[history_search, history_decision],
            outputs=[
                history_table,
                history_inspection_id,
                history_batch_id,
                history_summary,
            ],
        )
        history_view_button.click(
            fn=view_inspection_details,
            inputs=[history_inspection_id],
            outputs=[
                history_details,
                original_record_image,
                annotated_record_image,
            ],
        )
        individual_pdf_button.click(
            fn=download_inspection_pdf,
            inputs=[history_inspection_id],
            outputs=[individual_pdf_file],
        )
        batch_pdf_button.click(
            fn=download_batch_pdf,
            inputs=[history_batch_id],
            outputs=[batch_pdf_file],
        )

        # -----------------------------------------------------------------
        # Reports Event Bindings
        # -----------------------------------------------------------------
        def refresh_reports_dropdowns():
            records = database.list_inspections(limit=10_000, db_path=DATABASE_PATH)
            batches = database.list_batches(limit=10_000, db_path=DATABASE_PATH)
            ins_choices = [
                (f"{r['inspection_id']} · {r['image_filename']}", r["inspection_id"])
                for r in records
            ]
            batch_choices = [
                (f"{b['inspected_at']} · {b['batch_id']} ({b['total_images']} images)", b["batch_id"])
                for b in batches
            ]
            return (
                gr.Dropdown(
                    choices=ins_choices,
                    value=ins_choices[0][1] if ins_choices else None,
                    interactive=bool(ins_choices),
                ),
                gr.Dropdown(
                    choices=batch_choices,
                    value=batch_choices[0][1] if batch_choices else None,
                    interactive=bool(batch_choices),
                ),
            )

        reports_refresh_btn.click(
            fn=refresh_reports_dropdowns,
            inputs=None,
            outputs=[
                reports_inspection_id,
                reports_batch_id,
            ],
        )
        reports_individual_pdf_btn.click(
            fn=download_inspection_pdf,
            inputs=[reports_inspection_id],
            outputs=[reports_individual_pdf_file],
        )
        reports_batch_pdf_btn.click(
            fn=download_batch_pdf,
            inputs=[reports_batch_id],
            outputs=[reports_batch_pdf_file],
        )

        # -----------------------------------------------------------------
        # Page Initialization
        # -----------------------------------------------------------------
        def refresh_all_data(search_text: str = "", decision_filter: str = "ALL"):
            h_rows, ins_dd, batch_dd, h_summary = refresh_inspection_history(
                search_text, decision_filter
            )
            records = database.list_inspections(limit=10_000, db_path=DATABASE_PATH)
            batches = database.list_batches(limit=10_000, db_path=DATABASE_PATH)
            ins_choices = [
                (f"{r['inspection_id']} · {r['image_filename']}", r["inspection_id"])
                for r in records
            ]
            batch_choices = [
                (f"{b['inspected_at']} · {b['batch_id']} ({b['total_images']} images)", b["batch_id"])
                for b in batches
            ]
            reports_ins_dd = gr.Dropdown(
                choices=ins_choices,
                value=ins_choices[0][1] if ins_choices else None,
                interactive=bool(ins_choices),
            )
            reports_batch_dd = gr.Dropdown(
                choices=batch_choices,
                value=batch_choices[0][1] if batch_choices else None,
                interactive=bool(batch_choices),
            )
            kpi_html, d_chart, c_chart, r_rows = get_dashboard_analytics()
            return (
                h_rows,
                ins_dd,
                batch_dd,
                h_summary,
                reports_ins_dd,
                reports_batch_dd,
                kpi_html,
                d_chart,
                c_chart,
                r_rows,
            )

        demo.load(
            fn=refresh_all_data,
            inputs=[history_search, history_decision],
            outputs=[
                history_table,
                history_inspection_id,
                history_batch_id,
                history_summary,
                reports_inspection_id,
                reports_batch_id,
                dashboard_kpi_html,
                dashboard_decision_chart,
                dashboard_category_chart,
                dashboard_recent_table,
            ],
        )

        # Client-side 3D dynamic parallax tilt & interaction controller
        gr.HTML(
            """<script>
            (function init3DParallax() {
              function attachTiltListeners() {
                const tiltSelectors = [
                  '.metric-card',
                  '.hologram-card',
                  '.viewport-box',
                  '.specs-item',
                  '.pipeline-station'
                ];
                const cards = document.querySelectorAll(tiltSelectors.join(','));
                cards.forEach(card => {
                  if (card.dataset.tiltInitialized) return;
                  card.dataset.tiltInitialized = 'true';
                  
                  let rafId = null;
                  card.addEventListener('mousemove', (e) => {
                    const rect = card.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    const centerX = rect.width / 2;
                    const centerY = rect.height / 2;
                    const rotateX = ((y - centerY) / centerY) * -9.0;
                    const rotateY = ((x - centerX) / centerX) * 9.0;
                    
                    if (rafId) cancelAnimationFrame(rafId);
                    rafId = requestAnimationFrame(() => {
                      card.style.transform = `perspective(1000px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) translateZ(10px)`;
                    });
                  });
                  
                  card.addEventListener('mouseleave', () => {
                    if (rafId) cancelAnimationFrame(rafId);
                    card.style.transform = '';
                  });
                });
              }
              
              if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', attachTiltListeners);
              } else {
                attachTiltListeners();
              }
              setInterval(attachTiltListeners, 2000);
            })();
            </script>"""
        )

    return demo


_legacy_build_interface = build_interface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--yolo", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--mobilenet", type=Path, default=DEFAULT_MOBILENET)
    return parser.parse_args()


def main() -> None:
    global MODELS
    args = parse_args()
    database.initialize_database(DATABASE_PATH)
    print(f"Loading YOLO11 from {args.yolo}")
    print(f"Loading MobileNetV3 from {args.mobilenet}")
    MODELS = InspectionModels(args.yolo.resolve(), args.mobilenet.resolve())
    print(
        f"Models ready on {MODELS.device}; MobileNet threshold={MODELS.threshold:.6f}"
    )
    build_interface().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
        allowed_paths=[str(Path(tempfile.gettempdir()).resolve())],
    )


if __name__ == "__main__":
    main()
