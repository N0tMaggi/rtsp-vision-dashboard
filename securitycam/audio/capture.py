"""ffmpeg audio capture: command lines and one supervised subprocess.

ffmpeg decodes the camera's audio track (Tapo: PCM A-law / mu-law or AAC) and
writes 16 kHz mono s16le to stdout. The process is owned by a reader thread
that hands ~100 ms chunks to a callback. On Windows every ffmpeg is put into a
kill-on-close Job Object, so it dies with SecurityCam even after a crash; on
``stop()`` it is terminated, then killed, and always reaped (no zombies).

The RTSP URL carries the camera credentials: it only ever goes into argv,
never into a log line (``redact`` is applied to everything printed).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, List, Optional

from .levels import CHUNK_SAMPLES, SAMPLE_RATE

CHUNK_BYTES = CHUNK_SAMPLES * 2
RTSP_TIMEOUT_US = 5_000_000     # socket I/O timeout (ffmpeg >= 5: -timeout, microseconds)
STDERR_TAIL = 2048

# ffmpeg stderr -> status codes the UI understands
NO_AUDIO_RE = re.compile(r"matches no streams|does not contain any stream|Output file #0 does not contain|"
                         r"Output file does not contain any stream", re.I)
AUTH_RE = re.compile(r"\b401\b(?! *>=)|Unauthorized|authorization failed", re.I)
# harmless muxer chatter of some cameras (Tapo); kept out of the stderr tail so real errors stay visible
NOISE_RE = re.compile(r"non monotonically increasing dts|^\s*$", re.I)


def redact(text: str) -> str:
    """Hide passwords in URLs (rtsp://user:pass@host -> rtsp://user:***@host)."""
    return re.sub(r"(//[^:/@\s]*:)[^@/\s]*@", r"\1***@", text or "")


def ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def _output_args() -> List[str]:
    return ["-vn", "-sn", "-dn", "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]


def rtsp_command(url: str, ffmpeg: str = "ffmpeg") -> List[str]:
    """ffmpeg argv for the audio track of an RTSP stream (TCP, 5 s I/O timeout)."""
    return [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-timeout", str(RTSP_TIMEOUT_US),
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-i", url] + _output_args()


def _atempo_chain(speed: float) -> List[str]:
    """atempo factors whose product is ``speed`` (each within ffmpeg's 0.5..2.0)."""
    parts = []
    s = speed
    while s > 2.0:
        parts.append(2.0)
        s /= 2.0
    while s < 0.5:
        parts.append(0.5)
        s /= 0.5
    parts.append(s)
    return [f"atempo={p:.4f}" for p in parts if abs(p - 1.0) > 1e-3]


def file_command(path: str, position_s: float = 0.0, speed: float = 1.0, ffmpeg: str = "ffmpeg") -> List[str]:
    """ffmpeg argv that plays a video file's audio in real time from ``position_s``.

    ``-readrate`` paces input at playback speed; atempo turns it back into
    real-time 16 kHz audio so the analysis sees the same clock as the video.
    """
    speed = max(0.25, min(8.0, float(speed or 1.0)))
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error"]
    if position_s > 0.05:
        cmd += ["-ss", f"{position_s:.3f}"]
    cmd += ["-readrate", f"{speed:.4f}", "-i", path]
    out = _output_args()
    chain = _atempo_chain(speed)
    if chain:
        out = out[:-1] + ["-af", ",".join(chain)] + out[-1:]
    return cmd + out


# ----------------------------------------------------------------------------- Windows job object
class _Job:
    """Kill-on-close Job Object: children die when this process exits (Windows only)."""

    def __init__(self):
        self.handle = None
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class BASIC(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                            ("PerJobUserTimeLimit", ctypes.c_longlong),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class EXTENDED(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
            k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
            k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            handle = k32.CreateJobObjectW(None, None)
            if not handle:
                return
            info = EXTENDED()
            info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                return
            self._k32 = k32
            self.handle = handle
        except Exception:
            self.handle = None

    def assign(self, proc) -> None:
        if self.handle is None:
            return
        try:
            self._k32.AssignProcessToJobObject(self.handle, int(proc._handle))
        except Exception:
            pass


_job: Optional[_Job] = None
_job_lock = threading.Lock()


def _job_object() -> _Job:
    global _job
    with _job_lock:
        if _job is None:
            _job = _Job()
        return _job


# ----------------------------------------------------------------------------- process
class FfmpegReader:
    """One ffmpeg process + reader thread. ``on_chunk(bytes)`` gets CHUNK_BYTES at a time.

    ``on_exit(reader)`` runs once on the reader thread after the process ended
    (by itself or through ``stop()``); ``exit_code``, ``stderr_tail``,
    ``got_audio`` and ``stopped`` describe what happened.
    """

    def __init__(self, cmd: List[str], on_chunk: Callable[[bytes], None],
                 on_exit: Optional[Callable[["FfmpegReader"], None]] = None,
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen, name: str = "audio"):
        self.cmd = list(cmd)
        self.on_chunk = on_chunk
        self.on_exit = on_exit
        self.popen = popen
        self.name = name
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.stderr_tail = ""
        self.exit_code: Optional[int] = None
        self.got_audio = False
        self.stopped = False
        self.started_at = 0.0
        self.bytes_read = 0
        self._lock = threading.Lock()

    def start(self) -> "FfmpegReader":
        kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
                  "bufsize": 0}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        else:
            kwargs["start_new_session"] = True
        self.started_at = time.monotonic()
        self.proc = self.popen(self.cmd, **kwargs)
        _job_object().assign(self.proc)
        threading.Thread(target=self._drain_stderr, daemon=True, name=f"AudioErr-{self.name}").start()
        self.thread = threading.Thread(target=self._read, daemon=True, name=f"AudioRead-{self.name}")
        self.thread.start()
        return self

    @property
    def running(self) -> bool:
        return self.proc is not None and self.exit_code is None and not self.stopped

    def _drain_stderr(self) -> None:
        proc = self.proc
        try:
            for line in iter(proc.stderr.readline, b""):
                text = redact(line.decode("utf-8", "replace"))
                if NOISE_RE.search(text):
                    continue
                with self._lock:
                    self.stderr_tail = (self.stderr_tail + text)[-STDERR_TAIL:]
        except Exception:
            pass

    def _read(self) -> None:
        proc = self.proc
        buf = b""
        try:
            while not self.stopped:
                data = proc.stdout.read(CHUNK_BYTES - len(buf))
                if not data:
                    break
                buf += data
                if len(buf) < CHUNK_BYTES:
                    continue
                chunk, buf = buf, b""
                self.got_audio = True
                self.bytes_read += len(chunk)
                try:
                    self.on_chunk(chunk)
                except Exception as exc:     # analysis bugs must not kill the capture
                    print(f"[Sound] {self.name}: chunk handler failed: {exc}")
        except Exception:
            pass
        finally:
            self._reap()
            if self.on_exit is not None:
                try:
                    self.on_exit(self)
                except Exception as exc:
                    print(f"[Sound] {self.name}: exit handler failed: {exc}")

    def _reap(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception:
            pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        self.exit_code = proc.returncode if proc.returncode is not None else -1

    def stop(self, join: bool = True) -> None:
        """Terminate ffmpeg (then kill) and wait for the reader thread."""
        self.stopped = True
        proc = self.proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        t = self.thread
        if join and t is not None and t is not threading.current_thread():
            t.join(timeout=4.0)
            if t.is_alive() and proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass

    def stderr_text(self) -> str:
        with self._lock:
            return self.stderr_tail


def classify_exit(stderr: str, got_audio: bool) -> str:
    """ok | no_audio | auth | error, from an ended ffmpeg's stderr."""
    if NO_AUDIO_RE.search(stderr or ""):
        return "no_audio"
    if got_audio:
        return "ok"                 # the camera accepted the login; it just ended
    if AUTH_RE.search(stderr or ""):
        return "auth"
    return "ok" if got_audio else "error"


def env_ffmpeg() -> Optional[str]:
    """ffmpeg executable: SECURITYCAM_FFMPEG if set, else the one on PATH."""
    custom = os.getenv("SECURITYCAM_FFMPEG", "").strip()
    return custom or ffmpeg_path()
