"""Push alarm events to ntfy, Telegram and/or Discord (all optional, via .env).

Sending happens on one background thread fed by a bounded queue; the render
path never waits for the network. A cooldown per (camera, source) limits the
rate. Errors are logged with the channel name and error type only - URLs,
tokens and chat ids are never printed.
"""
from __future__ import annotations

import base64
import json
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Optional

from . import conf

TIMEOUT_S = 10.0
_queue: "queue.Queue[tuple[dict, Optional[bytes]]]" = queue.Queue(maxsize=32)
_last_sent: dict[tuple, float] = {}
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None


def channels() -> dict[str, bool]:
    return {
        "ntfy": bool(conf.NTFY_URL),
        "telegram": bool(conf.TELEGRAM_BOT_TOKEN and conf.TELEGRAM_CHAT_ID),
        "discord": bool(conf.DISCORD_WEBHOOK_URL),
    }


def _multipart(fields: dict[str, str], files: list[tuple[str, str, str, bytes]]) -> tuple[bytes, str]:
    boundary = f"----securitycam{uuid.uuid4().hex}"
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode()
        out += value.encode("utf-8") + b"\r\n"
    for name, filename, ctype, data in files:
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _http(url: str, data: bytes, headers: dict[str, str], method: str = "POST") -> int:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        resp.read(1024)
        return resp.status


def _header_value(text: str) -> str:
    """HTTP header values must be latin-1; ntfy understands RFC 2047 for UTF-8."""
    text = " ".join(text.split())
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        return "=?UTF-8?B?" + base64.b64encode(text.encode("utf-8")).decode("ascii") + "?="


def _title(event: dict) -> str:
    return f"SecurityCam alarm · {event.get('camera_label') or event.get('camera')}"


def _text(event: dict) -> str:
    when = time.strftime("%H:%M:%S", time.localtime(event.get("ts", time.time())))
    return f"{event.get('message', 'Alarm')} ({when})"


def send_ntfy(event: dict, image: Optional[bytes]) -> int:
    headers = {"Title": _header_value(_title(event)), "Tags": "rotating_light", "Priority": "high"}
    if conf.NTFY_TOKEN:
        headers["Authorization"] = f"Bearer {conf.NTFY_TOKEN}"
    if image:
        headers["Message"] = _header_value(_text(event))
        headers["Filename"] = "snapshot.jpg"
        return _http(conf.NTFY_URL, image, headers, "PUT")
    headers["Content-Type"] = "text/plain; charset=utf-8"
    return _http(conf.NTFY_URL, _text(event).encode("utf-8"), headers)


def send_telegram(event: dict, image: Optional[bytes]) -> int:
    base = f"{conf.TELEGRAM_API}/bot{conf.TELEGRAM_BOT_TOKEN}"
    caption = f"{_title(event)}\n{_text(event)}"
    if image:
        body, ctype = _multipart({"chat_id": conf.TELEGRAM_CHAT_ID, "caption": caption[:1000]},
                                 [("photo", "snapshot.jpg", "image/jpeg", image)])
        return _http(f"{base}/sendPhoto", body, {"Content-Type": ctype})
    body = json.dumps({"chat_id": conf.TELEGRAM_CHAT_ID, "text": caption}).encode("utf-8")
    return _http(f"{base}/sendMessage", body, {"Content-Type": "application/json"})


def send_discord(event: dict, image: Optional[bytes]) -> int:
    payload = {"content": f"**{_title(event)}**\n{_text(event)}", "username": "SecurityCam"}
    if image:
        payload["embeds"] = [{"image": {"url": "attachment://snapshot.jpg"}, "color": 0xFF5A4E}]
        body, ctype = _multipart({"payload_json": json.dumps(payload)},
                                 [("files[0]", "snapshot.jpg", "image/jpeg", image)])
        return _http(conf.DISCORD_WEBHOOK_URL, body, {"Content-Type": ctype})
    return _http(conf.DISCORD_WEBHOOK_URL, json.dumps(payload).encode("utf-8"),
                 {"Content-Type": "application/json"})


_SENDERS = {"ntfy": send_ntfy, "telegram": send_telegram, "discord": send_discord}


def deliver(event: dict, image: Optional[bytes]) -> dict[str, str]:
    """Send synchronously to every configured channel; returns {channel: "ok" | error type}."""
    result: dict[str, str] = {}
    for name, enabled in channels().items():
        if not enabled:
            continue
        try:
            status = _SENDERS[name](event, image)
            result[name] = "ok" if 200 <= status < 300 else f"http_{status}"
        except urllib.error.HTTPError as exc:
            result[name] = f"http_{exc.code}"
        except Exception as exc:  # network errors, timeouts, ...
            result[name] = exc.__class__.__name__
        if result[name] != "ok":
            print(f"[Notify] {name} failed: {result[name]}")
    return result


def _worker() -> None:
    while True:
        event, image = _queue.get()
        try:
            deliver(event, image)
        except Exception as exc:  # pragma: no cover
            print(f"[Notify] worker error: {exc.__class__.__name__}")


def _ensure_thread() -> None:
    global _thread
    with _lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_worker, name="AlertNotifier", daemon=True)
            _thread.start()


def submit(event: dict, image: Optional[bytes], key: tuple, force: bool = False) -> bool:
    """Queue a notification unless ``key`` is in cooldown or nothing is configured."""
    if not any(channels().values()):
        return False
    now = time.monotonic()
    with _lock:
        last = _last_sent.get(key)
        if not force and last is not None and now - last < conf.NOTIFY_COOLDOWN_S:
            return False
        _last_sent[key] = now
    _ensure_thread()
    try:
        _queue.put_nowait((event, image))
    except queue.Full:
        print("[Notify] queue full, dropping notification")
        return False
    return True
