"""Geometry helpers for the Face V2 segmentation pipeline.

This module contains ellipse fitting and partial-arc reconstruction. It is kept
separate from model inference so the geometry can be reviewed and tested
without loading the neural-network checkpoint.
"""

from __future__ import annotations

import cv2
import numpy as np


def fit_face_geometry(points):
    """
    Fit an ellipse to the YOLO segmentation polygon.

    Returns the geometry fields expected by app.py:

        cx
        cy
        r
        rx
        ry
        angle
        ratio
    """

    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

    if len(points) < 5:
        return None

    (cx, cy), (axis_1, axis_2), angle = cv2.fitEllipse(points)

    rx = float(axis_1 / 2.0)
    ry = float(axis_2 / 2.0)

    r = (rx + ry) / 2.0

    ratio = max(rx, ry) / max(
        1e-6,
        min(rx, ry),
    )

    return {
        "cx": float(cx),
        "cy": float(cy),
        "r": float(r),
        "rx": rx,
        "ry": ry,
        "angle": float(angle),
        "ratio": float(ratio),
    }


def fit_face_geometry_fallback(points):
    """
    Fallback geometry when the segmentation has too few points
    for cv2.fitEllipse().
    """

    points = np.asarray(points, dtype=np.float32)

    if len(points) < 3:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(points)

    r = float(radius)

    return {
        "cx": float(cx),
        "cy": float(cy),
        "r": r,
        "rx": r,
        "ry": r,
        "angle": 0.0,
        "ratio": 1.0,
    }


def _circle_from_three_points(a, b, c):
    """Return the circumcircle through three 2-D points, or None if degenerate."""
    ax, ay = map(float, a)
    bx, by = map(float, b)
    cx, cy = map(float, c)
    determinant = 2.0 * (
        ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    )
    if abs(determinant) < 1e-6:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    center_x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / determinant
    center_y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / determinant
    radius = float(np.hypot(ax - center_x, ay - center_y))
    if not np.isfinite([center_x, center_y, radius]).all() or radius <= 1.0:
        return None
    return float(center_x), float(center_y), radius


def _fit_circle_least_squares(points):
    """Algebraic least-squares circle fit used to refine RANSAC inliers."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return None
    x, y = points[:, 0], points[:, 1]
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    target = x * x + y * y
    try:
        solution, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    center_x, center_y, constant = solution
    radius_squared = constant + center_x * center_x + center_y * center_y
    if radius_squared <= 1.0 or not np.isfinite(radius_squared):
        return None
    return float(center_x), float(center_y), float(np.sqrt(radius_squared))


def _angular_coverage_degrees(points, center_x, center_y):
    """Measure how much of a circle's circumference the inliers span."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    angles = np.sort(np.mod(np.degrees(np.arctan2(
        points[:, 1] - center_y, points[:, 0] - center_x
    )), 360.0))
    if len(angles) < 3:
        return 0.0
    gaps = np.diff(np.concatenate((angles, angles[:1] + 360.0)))
    return float(360.0 - np.max(gaps))


def reconstruct_round_face_from_fragment(contour, image_shape):
    """
    Infer a full circular reference from a broken cutter's visible outer arc.

    Returns None unless a broad arc supports the fit. The visible segmentation
    is never discarded; callers retain it separately from this full reference.
    """
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(points) < 12:
        return None

    # Resample uniformly by physical distance along the contour.
    MAX_POINTS = 1000

    # Remove consecutive duplicate points.
    if len(points) > 1:
        step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep = np.concatenate(([True], step_lengths > 1e-6))
        points = points[keep]

    if len(points) < 12:
        return None

    # Include the closing segment if the contour is not already closed.
    if np.linalg.norm(points[0] - points[-1]) > 1e-6:
        points = np.vstack([points, points[0]])

    # Cumulative distance along the contour.
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative[-1]

    if total_length <= 1e-6:
        return None

    # Sample at equal physical-distance intervals.
    sample_count = min(MAX_POINTS, max(12, len(points) - 1))
    targets = np.linspace(0.0, total_length, sample_count, endpoint=False)

    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    points = np.column_stack((x, y))

    height, width = image_shape[:2]
    tolerance_floor = max(2.0, min(height, width) * 0.0015)
    rng = np.random.default_rng(1729)
    best = None

    for _ in range(min(900, max(250, len(points) * 3))):
        indices = rng.choice(len(points), size=3, replace=False)
        circle = _circle_from_three_points(points[indices[0]], points[indices[1]], points[indices[2]])
        if circle is None:
            continue
        center_x, center_y, radius = circle
        if radius < 8.0 or radius > max(height, width) * 0.75:
            continue
        if not (-0.25 * width <= center_x <= 1.25 * width):
            continue
        if not (-0.25 * height <= center_y <= 1.25 * height):
            continue

        distances = np.hypot(points[:, 0] - center_x, points[:, 1] - center_y)
        inliers = points[np.abs(distances - radius) <= max(tolerance_floor, radius * 0.018)]
        if len(inliers) < max(12, int(len(points) * 0.12)):
            continue

        refined = _fit_circle_least_squares(inliers)
        if refined is None:
            continue
        center_x, center_y, radius = refined
        distances = np.hypot(points[:, 0] - center_x, points[:, 1] - center_y)
        tolerance = max(tolerance_floor, radius * 0.018)
        inliers = points[np.abs(distances - radius) <= tolerance]
        coverage = _angular_coverage_degrees(inliers, center_x, center_y)
        if coverage < 130.0 or len(inliers) < max(12, int(len(points) * 0.12)):
            continue

        residual = float(np.median(np.abs(
            np.hypot(inliers[:, 0] - center_x, inliers[:, 1] - center_y) - radius
        )))
        support = len(inliers) / max(1, len(points))
        quality = support * (coverage / 360.0) / (
            1.0 + residual / max(1.0, radius * 0.01)
        )
        if best is None or quality > best["quality"]:
            best = {
                "cx": center_x, "cy": center_y, "r": radius,
                "coverage": coverage, "support": support,
                "residual": residual, "quality": quality,
            }

    if best is None:
        return None
    if best["support"] < 0.16 or best["residual"] > max(
        tolerance_floor * 1.5, best["r"] * 0.015
    ):
        return None

    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    full_circle = np.column_stack((
        best["cx"] + best["r"] * np.cos(angles),
        best["cy"] + best["r"] * np.sin(angles),
    )).astype(np.float32)
    return {
        "polygon": full_circle,
        "cx": float(best["cx"]),
        "cy": float(best["cy"]),
        "r": float(best["r"]),
        "coverage_degrees": float(best["coverage"]),
        "support": float(best["support"]),
        "residual": float(best["residual"]),
    }
