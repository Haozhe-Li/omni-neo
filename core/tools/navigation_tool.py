import os
from typing import Literal

import httpx

from core.utils.redis_cache import l1cache

# Public OSRM demo server (FOSSGIS-sponsored, router.project-osrm.org /
# routing.openstreetmap.de). Free but rate-limited (~1 req/s) and intended for
# "reasonable, non-commercial" use — override with OSRM_BASE_URL to point at a
# self-hosted or commercial OSRM-compatible instance instead.
OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org")

# OSRM demo server profile names differ from the mode names we expose.
_PROFILE_BY_MODE = {
    "driving": "driving",
    "walking": "foot",
    "cycling": "bike",
}

_MAX_STEPS = 12


def _format_distance(meters: float) -> str:
    if meters >= 1000:
        return f"{meters / 1000:.1f} km"
    return f"{round(meters)} m"


def _describe_step(step: dict) -> str:
    maneuver = step.get("maneuver", {})
    m_type = maneuver.get("type", "")
    modifier = maneuver.get("modifier", "")
    name = step.get("name") or "the road"
    distance = _format_distance(step.get("distance", 0))

    if m_type == "depart":
        text = f"Head {modifier + ' ' if modifier else ''}on {name}"
    elif m_type == "arrive":
        return "Arrive at destination"
    elif m_type in ("turn", "end of road"):
        text = f"Turn {modifier or 'onto'} onto {name}"
    elif m_type in ("roundabout", "rotary", "roundabout turn", "exit roundabout", "exit rotary"):
        text = f"At the roundabout, take the exit onto {name}"
    elif m_type == "fork":
        text = f"At the fork, keep {modifier or 'straight'} onto {name}"
    elif m_type == "merge":
        text = f"Merge onto {name}"
    elif m_type in ("on ramp", "off ramp", "ramp"):
        text = f"Take the ramp onto {name}"
    else:
        text = f"Continue onto {name}"

    return f"{text} ({distance})"


def _build_route_summary(routes: list[dict]) -> list[str]:
    steps = []
    for leg in routes[0].get("legs", []):
        steps.extend(leg.get("steps", []))

    descriptions = [_describe_step(step) for step in steps]

    if len(descriptions) <= _MAX_STEPS:
        return descriptions

    head = descriptions[:_MAX_STEPS - 4]
    tail = descriptions[-3:]
    skipped = len(descriptions) - len(head) - len(tail)
    return head + [f"... ({skipped} more turns) ..."] + tail


@l1cache(ttl=3600 * 24)
def get_navigation(
    origin_lat: float,
    origin_lng: float,
    destination_lat: float,
    destination_lng: float,
    mode: Literal["driving", "walking", "cycling"] = "driving",
) -> dict:
    """Get driving/walking/cycling directions between two points.

    Use this whenever the user (or the plan you're writing) needs to know how
    to get from one place to another by car, on foot, or by bike — e.g. "how
    do I drive from the hotel to the museum". Look up each point's
    latitude/longitude first (e.g. via `google_search_places`), then pass them
    here.

    Args:
        origin_lat: Latitude of the starting point.
        origin_lng: Longitude of the starting point.
        destination_lat: Latitude of the destination.
        destination_lng: Longitude of the destination.
        mode: Travel mode — "driving", "walking", or "cycling". Default "driving".

    Returns:
        dict: `distance_km`, `duration_min` (if available), and a short
        `route_summary` list of plain-language directions (not a full
        turn-by-turn breakdown). On failure, a dict with an `error` key.
    """
    profile = _PROFILE_BY_MODE.get(mode, "driving")
    coordinates = f"{origin_lng},{origin_lat};{destination_lng},{destination_lat}"
    url = f"{OSRM_BASE_URL}/route/v1/{profile}/{coordinates}"

    try:
        response = httpx.get(
            url,
            params={
                "steps": "true",
                "overview": "false",
                "geometries": "geojson",
                "alternatives": "false",
                "annotations": "false",
            },
            timeout=10,
        )
        data = response.json()
    except Exception as e:
        return {"error": f"Failed to get navigation: {str(e)}"}

    if data.get("code") != "Ok" or not data.get("routes"):
        return {"error": data.get("message") or data.get("code") or "No route found."}

    route = data["routes"][0]
    return {
        "mode": mode,
        "distance_km": round(route.get("distance", 0) / 1000, 1),
        "duration_min": round(route.get("duration", 0) / 60) if route.get("duration") is not None else None,
        "route_summary": _build_route_summary(data["routes"]),
    }
