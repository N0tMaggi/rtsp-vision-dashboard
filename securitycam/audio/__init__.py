"""Sound ESP: camera / video audio levels, spectrogram, YAMNet sound labels and sound alarms.

Audio never leaves this machine: ffmpeg decodes it locally, the analysis runs
locally (OpenVINO, Intel NPU by default) and only levels, spectrogram columns
and labels go to the browser; no audio is recorded or stored.

* ``levels``     - dBFS, meter, noise floor, loud-event detector, mel spectrum
* ``capture``    - ffmpeg command lines + supervised subprocess (Windows job object)
* ``classes``    - curated security labels over YAMNet's 521 AudioSet classes
* ``classifier`` - YAMNet on OpenVINO (compute workload "sound") + benchmark runner
* ``rules``      - opt-in sound alarm rules (persisted section ``sound``)
* ``engine``     - per-camera engine and the supervisor thread
* ``api``        - ``/api/sound`` blueprint

Integration points (all small, fail-safe): ``webapp.create_app`` registers
``api.blueprint``; ``live.build_topics`` adds the ``sound`` SSE topic via
``live_payload``; ``compute.assignments.WORKLOADS`` has the ``sound`` entry and
``compute.benchmark`` delegates to ``classifier.bench``; ``deps.ASSETS`` has
the model URL; ``config.DEFAULT_ESP_SETTINGS["sound"]`` is the toggle.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def live_payload(cs) -> Optional[Dict[str, Any]]:
    """SSE ``sound`` topic payload, None while sound is off for the camera."""
    if not cs.esp_settings.get("sound", False):
        return None
    from . import engine

    engine.ensure_started()
    return engine.payload(cs)
