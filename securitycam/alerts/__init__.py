"""Alarms & events: zone rules, tripwires, abandoned objects, event clips, notifications.

Submodules are imported explicitly (``from securitycam.alerts import engine``)
to keep this package free of import cycles with ``securitycam.esp``:

* ``conf``     - .env driven settings (data dir, clip/retention limits, channels)
* ``store``    - rules + tripwires persisted in ``data/alerts.json`` (atomic writes)
* ``engine``   - per-frame evaluation (called from the ESP pipeline, cheap)
* ``clips``    - per-camera ring buffer of rendered JPEGs + clip encoder
* ``events``   - event creation (thumbnail, metadata, clip), listing, retention
* ``notify``   - ntfy / Telegram / Discord senders (background queue, cooldown)
* ``routes``   - Flask blueprint with the HTTP API
"""
