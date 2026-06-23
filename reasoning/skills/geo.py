"""Geospatial — distances and bearings on the globe (great-circle), and DMS↔decimal degree
conversion. Pure stdlib (math), zero model calls. Distances in km (Earth radius 6371 km)."""

from __future__ import annotations

import math

from .base import register

_R = 6371.0  # mean Earth radius, km


@register("geo.haversine", "great-circle distance between two lat/long points (km)",
          [((0, 0, 0, 1), 111.19492664455873),
           ((40.7128, -74.006, 51.5074, -0.1278), 5570.222179737958)])
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _R * math.asin(math.sqrt(a))


@register("geo.bearing", "initial compass bearing from point 1 to point 2 (degrees, 0=N)",
          [((0, 0, 1, 0), 0.0), ((0, 0, 0, 1), 90.0)])
def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


@register("geo.dms_to_decimal", "degrees-minutes-seconds → decimal degrees",
          [((40, 30, 0), 40.5), ((-122, 19, 12), -122.32)])
def dms_to_decimal(degrees: float, minutes: float, seconds: float) -> float:
    return (-1 if degrees < 0 else 1) * (abs(degrees) + minutes / 60 + seconds / 3600)


@register("geo.decimal_to_dms", "decimal degrees → degrees-minutes-seconds string",
          [((40.5,), "40° 30' 0\""), ((-122.32,), "-122° 19' 12\"")])
def decimal_to_dms(dd: float) -> str:
    sign = "-" if dd < 0 else ""
    dd = abs(dd)
    d = int(dd)
    rem = (dd - d) * 60
    m = int(rem)
    s = round((rem - m) * 60)
    return f"{sign}{d}° {m}' {s}\""
