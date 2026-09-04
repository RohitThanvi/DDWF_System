"""Small geo helpers shared across services/routes."""
from __future__ import annotations


def validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox
    if not (-180 <= min_lon < max_lon <= 180):
        raise ValueError("Invalid longitude range in bbox.")
    if not (-90 <= min_lat < max_lat <= 90):
        raise ValueError("Invalid latitude range in bbox.")


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2


def lat_weight(lat_deg: float) -> float:
    """cos(lat) area-distortion weight, used by LatWeightedRMSE at train time
    and available here for any serving-side diagnostics."""
    import math

    return math.cos(math.radians(lat_deg))
