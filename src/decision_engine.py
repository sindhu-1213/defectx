"""Operational product decisions based on object-level classifier predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ObjectPrediction:
    class_name: str | None
    defect_probability: float | None


@dataclass(frozen=True)
class ObjectDecision:
    class_name: str
    predicted_class: str
    confidence: float | None
    status: str


@dataclass(frozen=True)
class ProductDecision:
    product_decision: str
    decision_explanation: str
    normal_objects: int
    defective_objects: int
    uncertain_objects: int
    classification_confidences: str
    max_classification_confidence: float | None
    object_decisions: tuple[ObjectDecision, ...]


def decide_product(
    predictions: Sequence[ObjectPrediction],
    classifier_threshold: float,
    decision_confidence_threshold: float,
    known_classes: set[str],
    processing_error: str | None = None,
) -> ProductDecision:
    """Apply the operational confidence rule to actual object predictions."""
    if not math.isfinite(decision_confidence_threshold) or not (
        0.0 <= decision_confidence_threshold <= 1.0
    ):
        raise ValueError("Decision confidence threshold must be between 0 and 1.")
    if not math.isfinite(classifier_threshold) or not (0.0 <= classifier_threshold <= 1.0):
        raise ValueError("Classifier threshold must be between 0 and 1.")

    normal = 0
    defective = 0
    uncertain = 0
    has_confident_defect = False
    score_details: list[str] = []
    uncertain_reasons: list[str] = []
    object_decisions: list[ObjectDecision] = []

    for index, prediction in enumerate(predictions, start=1):
        class_name = prediction.class_name
        probability = prediction.defect_probability
        if class_name is None or class_name not in known_classes:
            uncertain += 1
            uncertain_reasons.append(f"object {index} has an unknown or invalid class")
            score_details.append(f"object {index} ({class_name or 'unknown'}): unavailable")
            object_decisions.append(
                ObjectDecision(class_name or "Unknown", "UNKNOWN", None, "uncertain")
            )
            continue
        if probability is None or not math.isfinite(probability) or not (
            0.0 <= probability <= 1.0
        ):
            uncertain += 1
            uncertain_reasons.append(f"object {index} has a missing or invalid confidence")
            score_details.append(f"object {index} ({class_name}): unavailable")
            object_decisions.append(
                ObjectDecision(class_name, "UNKNOWN", None, "uncertain")
            )
            continue

        predicted_defective = probability >= classifier_threshold
        predicted_label = "DEFECTIVE" if predicted_defective else "NORMAL"
        predicted_confidence = probability if predicted_defective else 1.0 - probability
        score_details.append(
            f"object {index} ({class_name}): {predicted_label} "
            f"{predicted_confidence:.3f} confidence "
            f"(defect probability {probability:.3f})"
        )
        status = "uncertain"
        if predicted_confidence < decision_confidence_threshold:
            uncertain += 1
            uncertain_reasons.append(
                f"object {index} classification confidence "
                f"{predicted_confidence:.3f} is below the operational threshold"
            )
        elif predicted_defective:
            defective += 1
            has_confident_defect = True
            status = "defective"
        else:
            normal += 1
            status = "normal"
        object_decisions.append(
            ObjectDecision(class_name, predicted_label, predicted_confidence, status)
        )

    if has_confident_defect:
        decision = "FAIL"
        explanation = f"{defective} object(s) are confidently classified as defective."
        if uncertain_reasons:
            explanation += " Uncertainty also requires review: " + "; ".join(uncertain_reasons) + "."
        if processing_error:
            explanation += f" Processing issue: {processing_error}"
    elif processing_error:
        decision = "MANUAL REVIEW"
        explanation = f"Processing error prevented a reliable decision: {processing_error}"
    elif not predictions:
        decision = "MANUAL REVIEW"
        explanation = "No objects were detected; the product could not be assessed."
    elif uncertain_reasons:
        decision = "MANUAL REVIEW"
        explanation = "Manual review is required: " + "; ".join(uncertain_reasons) + "."
    else:
        decision = "PASS"
        explanation = f"All {normal} detected object(s) are confidently classified as normal."

    return ProductDecision(
        product_decision=decision,
        decision_explanation=explanation,
        normal_objects=normal,
        defective_objects=defective,
        uncertain_objects=uncertain,
        classification_confidences="; ".join(score_details),
        max_classification_confidence=max(
            (
                decision.confidence
                for decision in object_decisions
                if decision.confidence is not None
            ),
            default=None,
        ),
        object_decisions=tuple(object_decisions),
    )
