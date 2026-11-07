"""Curated security-relevant sound labels on top of YAMNet's 521 AudioSet classes.

Each curated label scores as the maximum of its AudioSet member classes
(YAMNet outputs independent per-class sigmoid scores). Indices refer to
``yamnet_label_list.txt`` (the AudioSet class map, shipped inside the TFLite
model); the names are kept next to them so a model with a different class
order is caught by ``check_names``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

NUM_CLASSES = 521


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    icon: str                         # lucide icon name used by the frontend
    members: Tuple[Tuple[int, str], ...]
    notable: bool = True              # shows a radar ping when it appears (speech / music do not)


CATEGORIES: Tuple[Category, ...] = (
    Category("speech", "Speech", "mic", (
        (0, "Speech"), (1, "Child speech, kid speaking"), (2, "Conversation"),
        (3, "Narration, monologue"), (12, "Whispering")), notable=False),
    Category("scream", "Shout / scream", "megaphone", (
        (6, "Shout"), (7, "Bellow"), (9, "Yell"), (10, "Children shouting"), (11, "Screaming"))),
    Category("glass_break", "Glass breaking", "glass", (
        (437, "Shatter"), (435, "Glass"))),
    Category("dog", "Dog bark", "dog", (
        (69, "Dog"), (70, "Bark"), (71, "Yip"), (72, "Howl"), (73, "Bow-wow"), (74, "Growling"))),
    Category("cat", "Cat", "cat", (
        (76, "Cat"), (78, "Meow"), (79, "Hiss"), (80, "Caterwaul"))),
    Category("knock", "Knock", "knock", (
        (353, "Knock"), (354, "Tap"))),
    Category("door", "Door", "door", (
        (348, "Door"), (351, "Sliding door"), (352, "Slam"))),
    Category("footsteps", "Footsteps", "footprints", (
        (48, "Walk, footsteps"), (46, "Run"), (47, "Shuffle"))),
    Category("alarm", "Alarm / siren", "siren", (
        (382, "Alarm"), (389, "Alarm clock"), (390, "Siren"), (391, "Civil defense siren"),
        (304, "Car alarm"), (316, "Emergency vehicle"), (317, "Police car (siren)"),
        (318, "Ambulance (siren)"), (319, "Fire engine, fire truck (siren)"), (392, "Buzzer"))),
    Category("smoke_alarm", "Smoke alarm", "smoke", (
        (393, "Smoke detector, smoke alarm"), (394, "Fire alarm"))),
    Category("baby_cry", "Baby crying", "baby", (
        (20, "Baby cry, infant cry"), (19, "Crying, sobbing"), (21, "Whimper"))),
    Category("bang", "Bang / gunshot-like", "bang", (
        (420, "Explosion"), (421, "Gunshot, gunfire"), (422, "Machine gun"), (423, "Fusillade"),
        (425, "Cap gun"), (426, "Fireworks"), (427, "Firecracker"), (428, "Burst, pop"),
        (430, "Boom"), (460, "Bang"))),
    Category("music", "Music", "music", (
        (132, "Music"), (133, "Musical instrument"), (24, "Singing")), notable=False),
    Category("vehicle", "Vehicle / horn", "car", (
        (294, "Vehicle"), (300, "Motor vehicle (road)"), (301, "Car"),
        (302, "Vehicle horn, car horn, honking"), (310, "Truck"), (312, "Air horn, truck horn"),
        (320, "Motorcycle"), (308, "Car passing by")), notable=False),
    Category("doorbell", "Doorbell", "bell", (
        (349, "Doorbell"), (350, "Ding-dong"))),
    Category("clap", "Applause / clap", "clap", (
        (58, "Clapping"), (62, "Applause"), (57, "Finger snapping"))),
)
BY_ID: Dict[str, Category] = {c.id: c for c in CATEGORIES}
IDS: Tuple[str, ...] = tuple(c.id for c in CATEGORIES)

_INDEX = [np.array([i for i, _ in c.members], np.int64) for c in CATEGORIES]


def check_names(labels: Sequence[str]) -> List[str]:
    """Mismatches between the curated table and a model's label list (empty = all good)."""
    bad = []
    if len(labels) != NUM_CLASSES:
        return [f"model has {len(labels)} classes, expected {NUM_CLASSES}"]
    for c in CATEGORIES:
        for i, name in c.members:
            if labels[i] != name:
                bad.append(f"{c.id}: class {i} is {labels[i]!r}, expected {name!r}")
    return bad


def curate(scores: np.ndarray) -> Dict[str, float]:
    """521 AudioSet scores -> {curated id: score 0..1}."""
    s = np.asarray(scores, np.float32).reshape(-1)
    if s.size != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} scores, got {s.size}")
    return {c.id: float(np.clip(s[idx].max(), 0.0, 1.0)) for c, idx in zip(CATEGORIES, _INDEX)}


def top(curated: Dict[str, float], min_score: float, n: int = 3) -> List[Tuple[str, float]]:
    items = [(k, v) for k, v in curated.items() if v >= min_score]
    items.sort(key=lambda kv: -kv[1])
    return items[:n]


def top_raw(scores: np.ndarray, labels: Optional[Sequence[str]], n: int = 1,
            skip: Iterable[int] = (494, 500, 501, 502, 503, 504, 507, 508)) -> List[Tuple[str, float]]:
    """Best raw AudioSet classes (for the "heard" detail line), skipping silence / room tone."""
    s = np.asarray(scores, np.float32).reshape(-1)
    skip = set(skip)
    out = []
    for i in np.argsort(-s):
        if int(i) in skip:
            continue
        name = labels[int(i)] if labels and int(i) < len(labels) else f"class {int(i)}"
        out.append((name, float(s[int(i)])))
        if len(out) >= n:
            break
    return out


def meta() -> List[Dict[str, object]]:
    return [{"id": c.id, "label": c.label, "icon": c.icon, "notable": c.notable} for c in CATEGORIES]
