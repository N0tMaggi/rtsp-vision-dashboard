"""Re-entry id memory (securitycam/recognition/reid.py): ids survive occlusions.

Synthetic frames: each "person" is a coloured rectangle (shirt colour on top,
trouser colour below); the tracker ids are fed in by hand, like Ultralytics
would hand out a new id after an occlusion.

Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from securitycam.recognition import reid  # noqa: E402

W, H = 1280, 720


def frame_with(people):
    """people: list of (box, shirt_bgr, trousers_bgr, hidden_from_y or None)."""
    f = np.full((H, W, 3), 90, np.uint8)
    f[::7, :] = 70                                    # some background texture
    for box, shirt, trousers, hidden in people:
        x1, y1, x2, y2 = (int(v) for v in box)
        mid = (y1 + y2) // 2
        f[y1:mid, x1:x2] = shirt
        f[mid:y2, x1:x2] = trousers
        if hidden is not None:                        # a grey car in front of the legs
            f[int(hidden):y2, x1 - 20:x2 + 20] = (128, 128, 128)
    return f


RED, BLUE, GREEN, YELLOW, BLACK = (40, 40, 220), (200, 60, 30), (40, 180, 40), (30, 220, 230), (20, 20, 20)


def box_at(x, top=300, w=60, h=160):
    return (x, top, x + w, top + h)


class ReentryTests(unittest.TestCase):
    def setUp(self):
        self.m = reid.ReentryMatcher()

    def feed(self, t, rows, people):
        return self.m.process(frame_with(people), rows, now=t)

    def walk(self, rid, x0, t0, t1, vx=150.0, fps=10, shirt=RED, trousers=BLUE, cls=0):
        out = {}
        t = t0
        while t <= t1 + 1e-9:
            x = x0 + vx * (t - t0)
            b = box_at(x)
            out = self.feed(t, [(rid, cls, b)], [(b, shirt, trousers, None)])
            t = round(t + 1 / fps, 4)
        return out, x

    def test_new_id_right_after_vanishing_keeps_old_id(self):
        # the old tracker id vanishes and a new one appears in the very next frame (< REENTRY_MIN_LOST_S)
        _, x = self.walk(5, 100, 0.0, 2.0)
        x += 15
        b = box_at(x, top=300, h=160)
        out = self.feed(2.1, [(9, 0, b)], [(b, RED, BLUE, 380)])     # half hidden behind a car
        self.assertEqual(out[9], 5)
        self.assertEqual(self.m.reentries, 1)

    def test_half_hidden_behind_car_after_gap(self):
        _, x = self.walk(3, 100, 0.0, 2.0)
        # 1.2 s behind the car, then only the upper body is visible, at the predicted spot
        x_pred = x + 150 * 1.2
        b = (x_pred, 300, x_pred + 60, 390)                        # box cut off at the car roof
        out = self.feed(3.2, [(11, 0, b)], [(b, RED, BLACK, None)])
        self.assertEqual(out[11], 3)

    def test_head_only_box_keeps_id(self):
        _, x = self.walk(4, 100, 0.0, 2.0)
        b = (x + 15, 300, x + 75, 330)                              # just the head above a car
        out = self.feed(2.1, [(12, 0, b)], [(b, RED, RED, None)])
        self.assertEqual(out[12], 4)

    def test_far_away_new_person_does_not_steal_id(self):
        self.walk(3, 100, 0.0, 2.0)
        b = box_at(1100)                                             # other side of the frame
        out = self.feed(2.5, [(20, 0, b)], [(b, GREEN, YELLOW, None)])
        self.assertEqual(out[20], 20)

    def test_different_looking_person_at_same_spot_is_not_merged(self):
        _, x = self.walk(3, 100, 0.0, 2.0)
        b = box_at(x + 15)
        out = self.feed(2.1, [(21, 0, b)], [(b, GREEN, YELLOW, None)])
        self.assertEqual(out[21], 21)

    def test_other_class_is_never_claimed(self):
        _, x = self.walk(3, 100, 0.0, 2.0)
        b = box_at(x + 15)
        out = self.feed(2.1, [(22, 2, b)], [(b, RED, BLUE, None)])
        self.assertEqual(out[22], 22)

    def test_two_people_side_by_side_ambiguous_stays_separate(self):
        # two identical-looking people vanish next to each other: a new id must not pick one at random
        t = 0.0
        while t <= 2.0:
            a, b = box_at(300 + 100 * t), box_at(380 + 100 * t)
            self.feed(t, [(1, 0, a), (2, 0, b)], [(a, RED, BLUE, None), (b, RED, BLUE, None)])
            t = round(t + 0.1, 4)
        nb = box_at(300 + 100 * 2.6 + 40)
        out = self.feed(2.6, [(30, 0, nb)], [(nb, RED, BLUE, None)])
        self.assertEqual(out[30], 30)

    def test_left_through_edge_and_back(self):
        _, x = self.walk(6, 300, 0.0, 2.0, vx=-150)                 # walks out on the left
        b = (0, 330, 60, 520)
        self.feed(2.1, [], [])
        out = self.feed(4.0, [(40, 0, b)], [(b, RED, BLUE, None)])  # back on the left, a bit lower
        self.assertEqual(out[40], 6)

    def test_appearance_only_reentry_still_works_later(self):
        self.walk(7, 100, 0.0, 2.0)
        self.feed(2.5, [], [])
        b = box_at(900)
        out = self.feed(30.0, [(50, 0, b)], [(b, RED, BLUE, None)])   # same look, far away, much later
        self.assertEqual(out[50], 7)

    def test_never_two_boxes_with_the_same_id(self):
        _, x = self.walk(8, 100, 0.0, 1.0)
        a, b = box_at(x + 15), box_at(x + 30)
        out = self.feed(1.1, [(60, 0, a), (61, 0, b)], [(a, RED, BLUE, None), (b, RED, BLUE, None)])
        self.assertNotEqual(out[60], out[61])

    def test_velocity_is_capped(self):
        st = reid._IdState(canonical=1, cls=0)
        st.seen = 1
        st.observe((0, 0, 50, 100), 0.0)
        st.seen = 2
        st.observe((5000, 0, 5050, 100), 0.1)
        self.assertLessEqual(abs(st.vel[0]), reid.MAX_SPEED_H * 100 + 1e-6)

    def test_remap_results_accepts_time(self):
        self.assertEqual(reid.remap_results(self.m, frame_with([]), [], now=1.0), 0)


if __name__ == "__main__":
    unittest.main()
