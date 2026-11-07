"""Tests for the Sound ESP (securitycam/audio): level / dBFS math, noise floor and
loud events, class curation, alarm rules + cooldown, ffmpeg command lines and
redaction, the capture lifecycle (no subprocess while disabled), the compute
workload, settings validation, the SSE topic and the /api/sound endpoints.

All audio is synthetic, subprocess.Popen and the classifier are faked; no
camera, no ffmpeg and no model are needed.

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="securitycam-sound-test-")
os.environ.setdefault("SECURITYCAM_DATA_DIR", _TMP)
os.environ.setdefault("SECURITYCAM_SAVE_DEBOUNCE", "0.05")
os.environ["SECURITYCAM_AUTO_INSTALL_DEPS"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from securitycam import config, live, state, streaming  # noqa: E402
from securitycam.audio import capture, classes, classifier, engine, levels, rules  # noqa: E402
from securitycam.compute import assignments, benchmark, devices  # noqa: E402
from securitycam.validation import sanitize_esp_patch  # noqa: E402
from securitycam.webapp import app  # noqa: E402

SR = levels.SAMPLE_RATE
URL = "rtsp://admin:s3cr%21t@192.0.2.50:554/stream1"


def tone(freq: float, amp: float, seconds: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1 - 1 / 32768) * 32768).astype("<i2").tobytes()


def chunks(x: np.ndarray):
    b = pcm(x)
    for i in range(0, len(b) - capture.CHUNK_BYTES + 1, capture.CHUNK_BYTES):
        yield b[i:i + capture.CHUNK_BYTES]


def make_cs(camera_id="snd", kind="rtsp", sound=True, url=URL, source=None):
    cs = state.CameraRuntime(camera_id=camera_id, label=camera_id.title(), ip="192.0.2.50",
                             stream_path="stream1", rtsp_url=url)
    cs.esp_settings = dict(config.DEFAULT_ESP_SETTINGS, sound=sound)
    cs.source_kind = kind
    cs.source = source
    return cs


def live_feed(grabbed=True, clients=1):
    return SimpleNamespace(clients=clients, stream=SimpleNamespace(stopped=False, grabbed=grabbed))


class FakeClassifier:
    def __init__(self):
        self.submitted = []
        self.labels = []
        self.dropped = []

    def submit(self, key, window, cb):
        self.submitted.append((key, window.copy()))

    def drop(self, key):
        self.dropped.append(key)

    def info(self):
        return {"status": "ready", "reason": "", "device": "npu", "ms": 1.6}


class FakeProc:
    """Popen stand-in: stdout yields ``data`` then EOF, stderr yields ``err``."""

    instances: list = []

    def __init__(self, cmd, data=b"", err=b"", rc=0, **kw):
        self.cmd = cmd
        self.kwargs = kw
        self.stdout = io.BytesIO(data)
        self.stderr = io.BytesIO(err)
        self.returncode = None
        self._rc = rc
        self.terminated = False
        self._handle = 0
        FakeProc.instances.append(self)

    def poll(self):
        if self.stdout.tell() >= len(self.stdout.getvalue()):
            self.returncode = self._rc if not self.terminated else -15
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = self._rc
        return self.returncode


def fake_popen(data=b"", err=b"", rc=0):
    calls = []

    def popen(cmd, **kw):
        calls.append(cmd)
        return FakeProc(cmd, data, err, rc, **kw)

    popen.calls = calls
    return popen


class RulesReset(unittest.TestCase):
    def setUp(self):
        rules.RULES.reset()
        engine.shutdown()

    def tearDown(self):
        rules.RULES.reset()
        engine.shutdown()


# ---------------------------------------------------------------------------- levels
class LevelTests(unittest.TestCase):
    def test_dbfs_of_reference_signals(self):
        square = np.sign(tone(1000, 1.0, 0.1)) * (1 - 1 / 32768)
        self.assertAlmostEqual(levels.rms_dbfs(square), 0.0, delta=0.05)
        self.assertAlmostEqual(levels.rms_dbfs(tone(1000, 1.0, 0.1)), -3.01, delta=0.05)
        self.assertAlmostEqual(levels.rms_dbfs(tone(1000, 0.1, 0.1)), -23.01, delta=0.05)
        self.assertEqual(levels.rms_dbfs(np.zeros(1600, np.float32)), levels.MIN_DB)
        self.assertAlmostEqual(levels.peak_dbfs(tone(440, 0.5, 0.1)), -6.02, delta=0.05)
        self.assertEqual(levels.db(0.0), levels.MIN_DB)

    def test_pcm_bytes_round_trip(self):
        x = levels.to_float(pcm(tone(500, 0.25, 0.1)))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.size, 1600)
        self.assertAlmostEqual(float(np.max(np.abs(x))), 0.25, delta=0.001)

    def test_meter_fast_attack_slow_release(self):
        m = levels.Meter()
        m.update(-20.0)
        self.assertGreater(m.value, -60.0)          # one step covers most of the jump up
        up = m.value
        m.update(-90.0)
        self.assertGreater(m.value, up - 0.5 * (up + 90))   # falls slower than it rose

    def test_noise_floor_drops_fast_and_rises_slowly(self):
        f = levels.NoiseFloor()
        for _ in range(50):
            f.update(-40.0)
        self.assertAlmostEqual(f.value, -40.0, delta=0.1)
        for _ in range(10):                         # 1 s of a loud sound
            f.update(-10.0)
        self.assertLess(f.value, -36.0)
        for _ in range(10):                         # quiet again: back down quickly
            f.update(-60.0)
        self.assertLess(f.value, -58.0)
        f.update(-10.0, frozen=True)
        self.assertLess(f.value, -58.0)

    def test_loud_event_needs_level_above_floor_for_min_time(self):
        d = levels.LoudDetector(threshold_db=20, min_ms=300)
        self.assertIsNone(d.update(-25.0, -50.0, 100))
        self.assertIsNone(d.update(-25.0, -50.0, 100))
        ev = d.update(-24.0, -50.0, 100)
        self.assertIsNotNone(ev)
        self.assertAlmostEqual(ev.above_db, 26.0)
        self.assertAlmostEqual(ev.duration_ms, 300.0)
        self.assertIsNone(d.update(-25.0, -50.0, 100))        # fires once per episode
        self.assertIsNone(d.update(-50.0, -50.0, 100))        # episode ends
        self.assertFalse(d.active)
        self.assertIsNone(d.update(-25.0, -50.0, 200))        # a short blip is not loud
        self.assertIsNone(d.update(-60.0, -50.0, 100))

    def test_loud_detector_ignores_quiet_absolute_levels(self):
        d = levels.LoudDetector(threshold_db=20, min_ms=100, min_level_db=-55)
        self.assertIsNone(d.update(-70.0, -95.0, 500))         # 25 dB above a dead-silent floor

    def test_mel_spectrum_puts_a_tone_in_the_right_band(self):
        m = levels.MelSpectrum(n_bands=32)
        low, high = m.band_db(tone(300, 0.5, 0.1)), m.band_db(tone(4000, 0.5, 0.1))
        self.assertLess(int(np.argmax(low)), int(np.argmax(high)))
        self.assertAlmostEqual(float(np.max(m.band_db(tone(1000, 1.0, 0.1)))), 0.0, delta=1.5)
        col = m.column(tone(1000, 0.5, 0.1))
        self.assertEqual((col.dtype, col.size), (np.uint8, 32))
        self.assertEqual(int(m.column(np.zeros(1600, np.float32)).max()), 0)


# ---------------------------------------------------------------------------- classes
class ClassTests(unittest.TestCase):
    def test_curated_set_covers_the_security_labels(self):
        want = {"speech", "scream", "glass_break", "dog", "cat", "knock", "door", "footsteps", "alarm",
                "smoke_alarm", "baby_cry", "bang", "music", "vehicle", "doorbell", "clap"}
        self.assertEqual(set(classes.IDS), want)
        self.assertEqual(len(classes.IDS), len(set(classes.IDS)))
        for c in classes.CATEGORIES:
            self.assertTrue(c.members and c.icon and c.label)
            self.assertTrue(all(0 <= i < classes.NUM_CLASSES for i, _ in c.members))

    def test_curate_takes_the_max_member_score(self):
        s = np.zeros(521, np.float32)
        s[437] = 0.8      # Shatter
        s[435] = 0.3      # Glass
        s[70] = 0.6       # Bark
        cur = classes.curate(s)
        self.assertAlmostEqual(cur["glass_break"], 0.8, places=5)
        self.assertAlmostEqual(cur["dog"], 0.6, places=5)
        self.assertEqual(cur["cat"], 0.0)
        self.assertEqual([k for k, _ in classes.top(cur, 0.5)], ["glass_break", "dog"])
        with self.assertRaises(ValueError):
            classes.curate(np.zeros(10))

    def test_check_names_against_a_label_list(self):
        names = [f"class {i}" for i in range(521)]
        for c in classes.CATEGORIES:
            for i, n in c.members:
                names[i] = n
        self.assertEqual(classes.check_names(names), [])
        names[437] = "Something else"
        self.assertTrue(any("437" in m for m in classes.check_names(names)))
        self.assertTrue(classes.check_names(names[:10]))

    def test_check_names_against_the_real_model_if_downloaded(self):
        path = config.MODELS_DIR / classifier.MODEL_FILE
        if not path.is_file():
            self.skipTest("YAMNet model not downloaded")
        self.assertEqual(classes.check_names(classifier.labels_from_model(path)), [])

    def test_top_raw_skips_silence(self):
        s = np.zeros(521, np.float32)
        s[494] = 0.9      # Silence
        s[69] = 0.4
        names = ["x"] * 521
        names[69] = "Dog"
        self.assertEqual(classes.top_raw(s, names, 1), [("Dog", 0.4000000059604645)])


# ---------------------------------------------------------------------------- rules
class RuleTests(RulesReset):
    def test_everything_is_off_by_default(self):
        cfg = rules.RULES.config()
        self.assertFalse(cfg["loud"]["enabled"])
        self.assertFalse(any(r["enabled"] for r in cfg["classes"].values()))
        self.assertEqual(set(cfg["classes"]), set(classes.IDS))
        self.assertFalse(rules.RULES.any_enabled())

    def test_patch_validation(self):
        clean = rules.validate_patch({"classes": {"glass_break": {"enabled": True, "threshold": 2.0}},
                                      "loud": {"db_above_floor": 1, "min_ms": 99999}})
        self.assertEqual(clean["classes"]["glass_break"]["threshold"], 0.99)
        self.assertEqual(clean["loud"]["db_above_floor"], 6.0)
        self.assertEqual(clean["loud"]["min_ms"], 5000)
        for bad in ({}, [], {"classes": {"lion": {"enabled": True}}}, {"classes": {"dog": {"enabled": "yes"}}},
                    {"classes": {"dog": {"color": 1}}}, {"loud": {"threshold": float("nan")}},
                    {"display_threshold": True}, {"nope": 1}):
            with self.assertRaises(rules.RuleError, msg=repr(bad)):
                rules.validate_patch(bad)

    def test_threshold_and_cooldown(self):
        rules.RULES.apply({"classes": {"glass_break": {"enabled": True, "threshold": 0.5, "cooldown_s": 30}}})
        cur = {"glass_break": 0.4, "dog": 0.9}
        self.assertEqual(rules.RULES.check_classes("cam", cur, now=100.0), [])     # below threshold, dog off
        cur["glass_break"] = 0.7
        self.assertEqual(rules.RULES.check_classes("cam", cur, now=101.0), [("glass_break", 0.7, True)])
        self.assertEqual(rules.RULES.check_classes("cam", cur, now=120.0), [])     # cooling down
        self.assertEqual(len(rules.RULES.check_classes("other", cur, now=120.0)), 1)   # per camera
        self.assertEqual(len(rules.RULES.check_classes("cam", cur, now=131.5)), 1)

    def test_loud_rule(self):
        self.assertIsNone(rules.RULES.check_loud("cam", now=1.0))
        rules.RULES.apply({"loud": {"enabled": True, "cooldown_s": 10, "notify": False}})
        self.assertIs(rules.RULES.check_loud("cam", now=1.0), False)
        self.assertIsNone(rules.RULES.check_loud("cam", now=5.0))
        self.assertIs(rules.RULES.check_loud("cam", now=12.0), False)

    def test_persistence_round_trip(self):
        rules.RULES.apply({"classes": {"dog": {"enabled": True, "threshold": 0.6}}, "display_threshold": 0.3})
        dumped = rules.RULES.dump()
        rules.RULES.reset()
        clean, warnings = rules.Rules.validate(dict(dumped, classes=dict(dumped["classes"], lion={"enabled": True})))
        self.assertTrue(any("lion" in w for w in warnings))
        rules.RULES.load(clean)
        cfg = rules.RULES.config()
        self.assertTrue(cfg["classes"]["dog"]["enabled"])
        self.assertEqual(cfg["classes"]["dog"]["threshold"], 0.6)
        self.assertEqual(cfg["display_threshold"], 0.3)
        with self.assertRaises(ValueError):
            rules.Rules.validate([])

    def test_emit_goes_through_the_event_store(self):
        cs = make_cs()
        with mock.patch("securitycam.alerts.events.events.trigger", return_value=True) as trig:
            self.assertTrue(rules.emit(cs, "glass_break", "Glass breaking", "Glass breaking heard (80%)", True))
        ev = trig.call_args[0][0]
        self.assertEqual(ev["kind"], "sound:glass_break")
        self.assertEqual(ev["camera"], "snd")
        self.assertTrue(ev["notify"])


# ---------------------------------------------------------------------------- engine analysis
class EngineAnalysisTests(RulesReset):
    def make(self):
        self.clf = FakeClassifier()
        return engine.SoundEngine(make_cs(), popen=fake_popen(), classifier=self.clf)

    def test_levels_spectrogram_and_classification_cadence(self):
        eng = self.make()
        for c in chunks(tone(1000, 0.3, 2.0)):
            eng.feed(c)
        p = eng.payload()
        self.assertEqual(p["seq"], 20)
        self.assertAlmostEqual(p["level"], levels.rms_dbfs(tone(1000, 0.3, 0.1)), delta=0.2)
        self.assertEqual(len(p["cols"]) * 3 // 4, engine.COLS_KEPT * 32)      # base64 of 6 x 32 bytes
        # first window once 0.975 s are buffered, then every 0.5 s
        self.assertEqual(len(self.clf.submitted), 3)
        self.assertEqual(self.clf.submitted[0][1].size, classifier.WINDOW_SAMPLES)

    def test_loud_event_recorded_and_alarm_only_when_armed(self):
        eng = self.make()
        quiet, loud = tone(200, 0.002, 3.0), tone(2000, 0.5, 1.0)
        with mock.patch.object(engine, "emit") as emit:
            for c in chunks(np.concatenate([quiet, loud])):
                eng.feed(c)
            self.assertEqual([e["kind"] for e in eng.payload()["events"]], ["loud"])
            self.assertFalse(eng.payload()["events"][0]["alarm"])
            emit.assert_not_called()
            rules.RULES.apply({"loud": {"enabled": True}})
            for c in chunks(np.concatenate([quiet, loud])):
                eng.feed(c)
            emit.assert_called_once()
            self.assertEqual(emit.call_args[0][1], "loud")
        self.assertTrue(eng.payload()["events"][-1]["alarm"])

    def test_scores_make_labels_pings_and_alarms(self):
        eng = self.make()
        s = np.zeros(521, np.float32)
        s[437], s[0] = 0.9, 0.7      # Shatter, Speech
        with mock.patch.object(engine, "emit") as emit:
            eng.on_scores(s, 1.6, "npu")
            emit.assert_not_called()
            p = eng.payload()
            self.assertEqual([l["id"] for l in p["labels"]], ["glass_break", "speech"])
            self.assertEqual([e["kind"] for e in p["events"]], ["glass_break"])   # notable ping, speech is not
            rules.RULES.apply({"classes": {"glass_break": {"enabled": True, "threshold": 0.8}}})
            eng.on_scores(s, 1.6, "npu")
            emit.assert_called_once()
            self.assertEqual(emit.call_args[0][1], "glass_break")
        self.assertTrue(eng.payload()["events"][-1]["alarm"])

    def test_labels_go_stale(self):
        now = [100.0]
        eng = engine.SoundEngine(make_cs(), popen=fake_popen(), clock=lambda: now[0], classifier=FakeClassifier())
        s = np.zeros(521, np.float32)
        s[70] = 0.9
        eng.on_scores(s, 1.0, "cpu")
        self.assertTrue(eng.payload()["labels"])
        now[0] += engine.LABELS_STALE_S + 0.1
        self.assertEqual(eng.payload()["labels"], [])


# ---------------------------------------------------------------------------- ffmpeg commands
class CommandTests(unittest.TestCase):
    def test_rtsp_command(self):
        cmd = capture.rtsp_command(URL, "ffmpeg")
        joined = " ".join(cmd)
        for part in ("-rtsp_transport tcp", "-timeout 5000000", "-vn", "-map 0:a:0", "-ac 1", "-ar 16000",
                     "-f s16le", "pipe:1", "-nostdin"):
            self.assertIn(part, joined)
        self.assertEqual(cmd[cmd.index("-i") + 1], URL)

    def test_redaction(self):
        self.assertEqual(capture.redact(URL), "rtsp://admin:***@192.0.2.50:554/stream1")
        self.assertEqual(capture.redact("Error opening rtsp://u:p@h/x: 401"), "Error opening rtsp://u:***@h/x: 401")
        self.assertNotIn("s3cr", capture.redact(" ".join(capture.rtsp_command(URL))))

    def test_file_command_follows_position_and_speed(self):
        cmd = capture.file_command("C:/v/a.mp4", 12.5, 1.0, "ffmpeg")
        self.assertEqual(cmd[cmd.index("-ss") + 1], "12.500")
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))
        self.assertNotIn("-af", cmd)
        self.assertNotIn("-ss", capture.file_command("a.mp4", 0.0))
        fast = capture.file_command("a.mp4", 0.0, 4.0)
        self.assertEqual(fast[fast.index("-readrate") + 1], "4.0000")
        self.assertEqual(fast[fast.index("-af") + 1], "atempo=2.0000,atempo=2.0000")
        slow = capture.file_command("a.mp4", 0.0, 0.25)
        self.assertEqual(slow[slow.index("-af") + 1], "atempo=0.5000,atempo=0.5000")

    def test_exit_classification(self):
        self.assertEqual(capture.classify_exit("Stream map '0:a:0' matches no streams.", False), "no_audio")
        self.assertEqual(capture.classify_exit("method DESCRIBE failed: 401 Unauthorized", False), "auth")
        self.assertEqual(capture.classify_exit("Connection timed out", False), "error")
        self.assertEqual(capture.classify_exit("", True), "ok")
        # seen on a real Tapo stream: numbers containing 401 are not a login failure
        dts = "Application provided invalid, non monotonically increasing dts to muxer: 140116 >= 124010"
        self.assertEqual(capture.classify_exit(dts, False), "error")
        self.assertEqual(capture.classify_exit("Server returned 401 Unauthorized", True), "ok")

    def test_reader_delivers_whole_chunks_and_reaps(self):
        data = pcm(tone(440, 0.2, 0.35))           # 3.5 chunks: the partial tail is dropped
        got, exited = [], threading.Event()
        popen = fake_popen(data=data, err=b"some warning rtsp://u:pw@h/x\n")
        r = capture.FfmpegReader(["ffmpeg"], got.append, lambda rd: exited.set(), popen=popen, name="t").start()
        self.assertTrue(exited.wait(3))
        self.assertEqual(len(got), 3)
        self.assertTrue(all(len(c) == capture.CHUNK_BYTES for c in got))
        self.assertTrue(r.got_audio)
        self.assertIsNotNone(r.exit_code)
        r.thread.join(1)
        time.sleep(0.05)
        self.assertNotIn(":pw@", r.stderr_text())
        kw = FakeProc.instances[-1].kwargs
        self.assertEqual(kw["stdin"], capture.subprocess.DEVNULL)
        if sys.platform == "win32":
            self.assertTrue(kw["creationflags"] & 0x08000000)    # CREATE_NO_WINDOW


# ---------------------------------------------------------------------------- lifecycle
class LifecycleTests(RulesReset):
    def setUp(self):
        super().setUp()
        self.popen = fake_popen(err=b"")
        self.patches = [mock.patch.object(engine, "POPEN", self.popen),
                        mock.patch.object(capture, "env_ffmpeg", return_value="ffmpeg"),
                        mock.patch.object(engine, "CLASSIFIER", FakeClassifier())]
        for p in self.patches:
            p.start()

    def tearDown(self):
        engine.shutdown()
        for p in self.patches:
            p.stop()
        super().tearDown()

    def run_tick(self, cs, feed=None, failures=None):
        feeds = {cs.camera_id: feed} if feed is not None else {}
        with mock.patch.dict(state.camera_states, {cs.camera_id: cs}, clear=True), \
                mock.patch.dict(streaming._feeds, feeds, clear=True), \
                mock.patch.dict(streaming._connect_failures, failures or {}, clear=True):
            engine.tick_all()

    def test_no_subprocess_while_sound_is_off(self):
        self.assertFalse(config.DEFAULT_ESP_SETTINGS["sound"])
        cs = make_cs(sound=False)
        self.run_tick(cs, live_feed())
        self.assertEqual(self.popen.calls, [])
        self.assertIsNone(engine.get(cs.camera_id))
        self.assertIsNone(engine.payload(cs))

    def test_no_subprocess_while_the_camera_is_not_viewed(self):
        cs = make_cs()
        self.run_tick(cs, None)
        self.run_tick(cs, live_feed(clients=0))
        self.assertEqual(self.popen.calls, [])
        self.assertEqual(engine.payload(cs)["state"], "idle")

    def test_waits_for_the_video_connection_and_its_backoff(self):
        cs = make_cs()
        self.run_tick(cs, live_feed(grabbed=False))
        self.assertEqual(self.popen.calls, [])
        self.run_tick(cs, live_feed(), failures={cs.camera_id: (2, time.monotonic() + 30)})
        self.assertEqual(self.popen.calls, [])
        self.assertEqual(engine.get(cs.camera_id).state, "waiting_video")

    def test_starts_ffmpeg_with_the_camera_url_and_stops_it(self):
        cs = make_cs()
        with mock.patch("builtins.print") as printed:
            self.run_tick(cs, live_feed())
        self.assertEqual(len(self.popen.calls), 1)
        self.assertIn(URL, self.popen.calls[0])
        self.assertNotIn("s3cr", " ".join(str(c) for c in printed.call_args_list))
        eng = engine.get(cs.camera_id)
        self.assertIsNotNone(eng)
        cs.esp_settings["sound"] = False
        self.run_tick(cs, live_feed())
        self.assertIsNone(engine.get(cs.camera_id))
        self.assertTrue(eng.closed)

    def test_failed_audio_backs_off(self):
        cs = make_cs()
        eng = engine.SoundEngine(cs, popen=self.popen, classifier=FakeClassifier())
        reader = SimpleNamespace(stopped=False, got_audio=False, started_at=time.monotonic(),
                                 stderr_text=lambda: "401 Unauthorized")
        with mock.patch("builtins.print"):
            eng._on_exit(reader)
        self.assertEqual(eng.state, "auth")
        self.assertGreaterEqual(eng.retry_at - time.monotonic(), engine.AUTH_BACKOFF_S - 1)
        reader.stderr_text = lambda: "Stream map '0:a:0' matches no streams."
        eng._on_exit(reader)
        self.assertEqual(eng.state, "no_audio")
        self.assertEqual(eng.payload()["message"], "No audio track on this camera.")
        with mock.patch.dict(streaming._feeds, {cs.camera_id: live_feed()}):
            eng.tick()
        self.assertEqual(self.popen.calls, [])          # no retry before the backoff ends

    def test_error_backoff_doubles(self):
        eng = engine.SoundEngine(make_cs(), popen=self.popen, classifier=FakeClassifier())
        reader = SimpleNamespace(stopped=False, got_audio=False, started_at=time.monotonic(),
                                 stderr_text=lambda: "Connection refused")
        delays = []
        with mock.patch("builtins.print"):
            for _ in range(6):
                eng._on_exit(reader)
                delays.append(round(eng.retry_at - time.monotonic()))
        self.assertEqual(delays[:4], [5, 10, 20, 40])
        self.assertEqual(delays[-1], 60)

    def test_file_source_follows_playback(self):
        pb = SimpleNamespace(cond=threading.Condition(), playing=False, ended=False, speed=1.0, cuts=0, frame=250, fps=25.0)
        src = SimpleNamespace(path="C:/videos/clip.mp4", playback=pb)
        cs = make_cs("video-abc", kind="file", url="video://abc", source=src)
        self.run_tick(cs, live_feed())
        self.assertEqual(self.popen.calls, [])          # paused: nothing runs
        self.assertEqual(engine.get(cs.camera_id).state, "paused")
        pb.playing = True
        self.run_tick(cs, live_feed())
        self.assertEqual(len(self.popen.calls), 1)
        cmd = self.popen.calls[0]
        self.assertEqual(cmd[cmd.index("-ss") + 1], "10.000")
        self.assertIn("C:/videos/clip.mp4", cmd)

    def test_url_sources_are_not_supported(self):
        cs = make_cs("video-url", kind="url", url="video://u")
        self.run_tick(cs, live_feed())
        self.assertEqual(self.popen.calls, [])
        self.assertEqual(engine.get(cs.camera_id).state, "unsupported")


# ---------------------------------------------------------------------------- compute workload
class WorkloadTests(unittest.TestCase):
    def setUp(self):
        assignments.reset()
        classifier._auto_failed.clear()

    def tearDown(self):
        assignments.reset()
        classifier._auto_failed.clear()

    def with_hw(self, npu=True):
        cache = {"openvino": {"installed": True, "version": "x", "devices": {}, "error": None},
                 "ov_map": {"npu": "NPU" if npu else None, "igpu": "GPU.0", "cpu": "CPU"}, "seconds": 0, "at": 0}
        return [mock.patch.object(devices, "openvino_installed", return_value=True),
                mock.patch.object(devices, "_cache", cache)]

    def test_registered_with_npu_igpu_cpu(self):
        self.assertIn("sound", assignments.WORKLOADS)
        self.assertEqual(assignments.WORKLOADS["sound"]["devices"], ("npu", "igpu", "cpu"))
        self.assertEqual(assignments.assignment("sound"), "auto")
        self.assertTrue(callable(getattr(benchmark, "_bench_sound", None)))

    def test_auto_is_npu_when_present_else_cpu(self):
        ps = self.with_hw(npu=True)
        for p in ps:
            p.start()
        try:
            self.assertEqual(classifier.resolve_device()[0], "npu")
            self.assertEqual(classifier.auto_label()["device"], "npu")
            classifier._auto_failed.add("npu")              # NPU failed at runtime -> CPU
            self.assertEqual(classifier.resolve_device()[0], "cpu")
            assignments.apply_patch({"workloads": {"sound": "igpu"}})
            self.assertEqual(classifier.resolve_device()[0], "igpu")
        finally:
            for p in ps:
                p.stop()
        ps = self.with_hw(npu=False)
        for p in ps:
            p.start()
        try:
            assignments.reset()
            self.assertEqual(classifier.resolve_device()[0], "cpu")
            self.assertEqual(classifier.auto_label()["device"], "cpu")
        finally:
            for p in ps:
                p.stop()

    def test_without_openvino_classification_is_unavailable(self):
        with mock.patch.object(devices, "openvino_installed", return_value=False):
            dev, reason = classifier.resolve_device()
        self.assertIsNone(dev)
        self.assertIn("OpenVINO", reason)

    def test_benchmark_runner_uses_the_sound_bench(self):
        job = {"workload": "sound", "rows": [{"device": "npu"}, {"device": "cpu"}]}
        rows = {}

        class Req:
            def infer(self, _):
                pass

        class Compiled:
            def create_infer_request(self):
                return Req()

        with mock.patch.object(classifier, "model_path", return_value=Path("yamnet.tflite")), \
                mock.patch("securitycam.compute.ov_runtime.compile_raw", return_value=(Compiled(), 0.1)):
            benchmark._default_runner(job, lambda dev, **kw: rows.setdefault(dev, {}).update(kw))
        self.assertEqual({rows["npu"]["status"], rows["cpu"]["status"]}, {"done"})
        self.assertIsInstance(rows["npu"]["ms"], float)


# ---------------------------------------------------------------------------- settings / live / API
class IntegrationTests(RulesReset):
    def setUp(self):
        super().setUp()
        self.client = app.test_client()

    def test_setting_validation(self):
        self.assertIs(config.DEFAULT_ESP_SETTINGS["sound"], False)
        self.assertEqual(sanitize_esp_patch({"sound": "yes"}), {"sound": True})
        self.assertEqual(sanitize_esp_patch({"sound": 0}), {"sound": False})
        self.assertEqual(sanitize_esp_patch({"sound": "maybe"}), {})

    def test_sse_topic_only_while_sound_is_on(self):
        cs = make_cs(sound=False)
        topic = next(t for t in live.build_topics(cs, False) if t.name == "sound")
        live._SNAPSHOTS.clear()
        self.assertIsNone(topic.poll())
        cs.esp_settings["sound"] = True
        live._SNAPSHOTS.clear()
        with mock.patch.object(engine, "ensure_started"):
            snap = topic.poll()
        self.assertIsNotNone(snap)
        self.assertEqual(snap.payload["state"], "idle")
        self.assertLess(len(snap.data), 600)          # small payload

    def test_config_api(self):
        r = self.client.get("/api/sound/config")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(len(d["classes"]), len(classes.IDS))
        self.assertIn("locally", d["privacy"])
        self.assertIn("Apache-2.0", d["model"]["license"])
        r = self.client.patch("/api/sound/config", json={"classes": {"scream": {"enabled": True, "threshold": 0.7}}})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["rules"]["classes"]["scream"]["enabled"])
        self.assertEqual(self.client.patch("/api/sound/config", json={"classes": {"x": {}}}).get_json()["error"],
                         "unknown_class")
        self.assertEqual(self.client.patch("/api/sound/config", data="a=1",
                                           content_type="application/x-www-form-urlencoded").status_code, 415)

    def test_live_api(self):
        self.assertEqual(self.client.get("/api/sound/nope").status_code, 404)
        cs = make_cs("sndapi", sound=False)
        with mock.patch.dict(state.camera_states, {"sndapi": cs}), mock.patch.object(engine, "ensure_started"):
            self.assertEqual(self.client.get("/api/sound/sndapi").get_json(), {"camera": "sndapi", "on": False})


if __name__ == "__main__":
    unittest.main()
