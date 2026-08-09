"""Canonical camera profiles used by the optional Viser camera panels."""

from __future__ import annotations

PROFILE_CAMERA_TOPICS: dict[str, dict[str, str]] = {
    "single_overhead": {
        "wrist": "/follower/image_raw",
        "overhead_1": "/static_camera_1/image_raw",
    },
    "dual_overhead": {
        "wrist": "/follower/image_raw",
        "overhead_1": "/static_camera_1/image_raw",
        "overhead_2": "/static_camera_2/image_raw",
    },
}


def camera_topics(camera_profile: str) -> dict[str, str]:
    """Return a copy of the canonical topic map for a required profile."""
    try:
        return dict(PROFILE_CAMERA_TOPICS[camera_profile])
    except KeyError as exc:
        choices = ", ".join(PROFILE_CAMERA_TOPICS)
        raise ValueError(
            f"camera_profile must be one of: {choices} (got {camera_profile!r})"
        ) from exc
