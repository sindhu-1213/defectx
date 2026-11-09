"""PDF report generation for saved DefectX inspection records."""

from __future__ import annotations

import html
import tempfile
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image as ReportImage,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#172554")
BLUE = colors.HexColor("#1d4ed8")
GREEN = colors.HexColor("#15803d")
RED = colors.HexColor("#b91c1c")
AMBER = colors.HexColor("#b45309")
PALE = colors.HexColor("#eff6ff")


def _output_path(prefix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=prefix, suffix=".pdf", delete=False
    ) as handle:
        return Path(handle.name)


def _text(value: Any) -> str:
    if value is None or value == "":
        return "Not available"
    return html.escape(str(value)).replace("\n", "<br/>")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "DefectXTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            "DefectXSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "DefectXSmall",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=9.5,
        )
    )
    styles.add(
        ParagraphStyle(
            "DefectXDecision",
            parent=styles["Heading2"],
            fontSize=15,
            leading=19,
            alignment=TA_CENTER,
        )
    )
    return styles


def _canvas_without_compression(*args, **kwargs):
    kwargs["pageCompression"] = 0
    return Canvas(*args, **kwargs)


def _metadata_table(rows: list[tuple[str, Any]], styles) -> Table:
    data = [
        [
            Paragraph(f"<b>{_text(label)}</b>", styles["BodyText"]),
            Paragraph(_text(value), styles["BodyText"]),
        ]
        for label, value in rows
    ]
    table = Table(data, colWidths=[1.8 * inch, 4.9 * inch], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), PALE),
                ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _image_flowable(path_value: str | None, max_width: float, max_height: float):
    if not path_value:
        return Paragraph("Image not available for this record.", _styles()["BodyText"])
    path = Path(path_value)
    if not path.is_file():
        return Paragraph("Image file is no longer available.", _styles()["BodyText"])
    image = ReportImage(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight, 1.0)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def _decision_color(decision: str):
    return {
        "PASS": GREEN,
        "FAIL": RED,
        "MANUAL REVIEW": AMBER,
    }.get(decision, NAVY)


def _object_table(record: dict[str, Any], styles) -> Table:
    data = [["Object", "Product class", "Classification", "Confidence", "Outcome"]]
    for index, detail in enumerate(record.get("object_details", []), start=1):
        confidence = detail.get("confidence")
        data.append(
            [
                str(index),
                Paragraph(_text(detail.get("class_name")), styles["DefectXSmall"]),
                Paragraph(_text(detail.get("predicted_class")), styles["DefectXSmall"]),
                "Not available" if confidence is None else f"{confidence:.3f}",
                Paragraph(_text(detail.get("status")), styles["DefectXSmall"]),
            ]
        )
    if len(data) == 1:
        data.append(["—", "No object-level predictions", "—", "—", "—"])
    table = LongTable(data, colWidths=[0.6 * inch, 1.45 * inch, 1.45 * inch, 1.0 * inch, 1.35 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def generate_inspection_pdf(
    record: dict[str, Any], output_path: Path | str | None = None
) -> str:
    path = Path(output_path) if output_path else _output_path("defectx_inspection_")
    styles = _styles()
    document = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=f"DefectX Inspection {record['inspection_id']}",
        author="DefectX",
    )
    story = [
        Paragraph("DefectX", styles["DefectXTitle"]),
        Paragraph("Digital Inspection Report", styles["DefectXSubtitle"]),
    ]
    decision = str(record["product_decision"])
    decision_box = Table(
        [[Paragraph(_text(decision), styles["DefectXDecision"])]],
        colWidths=[6.7 * inch],
    )
    decision_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("TEXTCOLOR", (0, 0), (-1, -1), _decision_color(decision)),
                ("BOX", (0, 0), (-1, -1), 1.2, _decision_color(decision)),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.extend(
        [
            decision_box,
            Spacer(1, 12),
            _metadata_table(
                [
                    ("Inspection ID", record["inspection_id"]),
                    ("Inspection time (UTC)", record["inspected_at"]),
                    ("Image filename", record["image_filename"]),
                    ("Product category", record["product_category"]),
                    ("Detected objects", record["total_objects"]),
                    ("Normal objects", record["normal_objects"]),
                    ("Defective objects", record["defective_objects"]),
                    ("Uncertain objects", record["uncertain_objects"]),
                    ("Decision explanation", record["decision_explanation"]),
                    ("YOLO11 time (ms)", record.get("yolo_ms")),
                    ("MobileNetV3 time (ms)", record.get("mobilenet_ms")),
                    ("Total inference time (ms)", record.get("total_inference_ms")),
                    (
                        "Decision confidence threshold",
                        record["decision_confidence_threshold"],
                    ),
                    ("Classification confidence detail", record.get("classification_confidences")),
                ],
                styles,
            ),
            Spacer(1, 14),
            Paragraph("Inspection images", styles["Heading2"]),
        ]
    )
    image_cells = [
        _image_flowable(record.get("original_image_path"), 3.15 * inch, 2.7 * inch),
        _image_flowable(record.get("annotated_image_path"), 3.15 * inch, 2.7 * inch),
    ]
    images = Table(
        [
            image_cells,
            [
                Paragraph("Original image", styles["DefectXSmall"]),
                Paragraph("Annotated detection image", styles["DefectXSmall"]),
            ],
        ],
        colWidths=[3.35 * inch, 3.35 * inch],
        hAlign="LEFT",
    )
    images.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LINEBELOW", (0, 1), (-1, 1), 0.4, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            images,
            Spacer(1, 14),
            Paragraph("Object-level classification", styles["Heading2"]),
            _object_table(record, styles),
            Spacer(1, 14),
            Paragraph(
                "<b>Human verification:</b> This is an AI-assisted inspection result. "
                "A qualified person should verify the image and findings; manual-review "
                "or uncertain cases require human assessment. This report does not "
                "represent automotive certification or factory deployment validation.",
                styles["BodyText"],
            ),
        ]
    )
    document.build(story, canvasmaker=_canvas_without_compression)
    return str(path.resolve())


def _batch_summary_table(batch: dict[str, Any], styles) -> Table:
    rate = (
        f"{100 * batch['defect_rate']:.2f}%"
        if batch.get("defect_rate") is not None
        else "N/A (no confidently classified objects)"
    )
    return _metadata_table(
        [
            ("Batch ID", batch["batch_id"]),
            ("Inspection date (UTC)", batch["inspected_at"]),
            ("Total images", batch["total_images"]),
            ("Successfully processed", batch["successful_images"]),
            ("PASS / FAIL / MANUAL REVIEW", f"{batch['pass_count']} / {batch['fail_count']} / {batch['review_count']}"),
            ("Processing errors", batch["processing_errors"]),
            ("Object defect rate", rate),
            (
                "Defect-rate calculation",
                f"{batch['defective_objects']} confidently defective objects / "
                f"{batch['classified_objects']} confidently classified normal and defective objects; "
                "uncertain objects and processing errors are excluded.",
            ),
        ],
        styles,
    )


def _batch_table(records: list[dict[str, Any]], styles) -> LongTable:
    small = styles["DefectXSmall"]
    headers = [
        "Inspection ID",
        "Image",
        "Category",
        "Total",
        "Normal",
        "Defect",
        "Uncertain",
        "Decision",
        "Total ms",
        "Decision explanation",
    ]
    data = [[Paragraph(f"<b>{_text(value)}</b>", small) for value in headers]]
    for record in records:
        data.append(
            [
                Paragraph(_text(record["inspection_id"]), small),
                Paragraph(_text(record["image_filename"]), small),
                Paragraph(_text(record["product_category"]), small),
                str(record["total_objects"]),
                str(record["normal_objects"]),
                str(record["defective_objects"]),
                str(record["uncertain_objects"]),
                Paragraph(_text(record["product_decision"]), small),
                "Not available"
                if record.get("total_inference_ms") is None
                else f"{record['total_inference_ms']:.2f}",
                Paragraph(_text(record["decision_explanation"]), small),
            ]
        )
    widths = [
        1.02 * inch,
        1.2 * inch,
        0.8 * inch,
        0.38 * inch,
        0.4 * inch,
        0.4 * inch,
        0.48 * inch,
        0.82 * inch,
        0.58 * inch,
        2.1 * inch,
    ]
    table = LongTable(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def generate_batch_pdf(
    batch: dict[str, Any],
    records: list[dict[str, Any]],
    output_path: Path | str | None = None,
) -> str:
    path = Path(output_path) if output_path else _output_path("defectx_batch_")
    styles = _styles()
    document = SimpleDocTemplate(
        str(path),
        pagesize=landscape(letter),
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title=f"DefectX Batch {batch['batch_id']}",
        author="DefectX",
    )
    story = [
        Paragraph("DefectX", styles["DefectXTitle"]),
        Paragraph("Consolidated Batch Inspection Report", styles["DefectXSubtitle"]),
        _batch_summary_table(batch, styles),
        Spacer(1, 14),
        Paragraph("Image-wise inspection records", styles["Heading2"]),
    ]
    story.append(
        _batch_table(records, styles)
        if records
        else Paragraph("No image inspection records are available for this batch.", styles["BodyText"])
    )
    story.extend(
        [
            Spacer(1, 14),
            Paragraph(
                "AI-assisted inspection summary. Review uncertain cases and verify "
                "findings with an appropriate human reviewer. This report does not "
                "represent automotive certification or factory deployment validation.",
                styles["BodyText"],
            ),
        ]
    )
    document.build(story, canvasmaker=_canvas_without_compression)
    return str(path.resolve())
