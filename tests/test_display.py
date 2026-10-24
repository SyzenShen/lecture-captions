import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import live  # noqa: E402


class FakeOut:
    def __init__(self):
        self.shown = []

    def partial(self, text):
        self.shown.append(text)

    def _clear_partial(self):
        self.shown.append(None)


class DisplayTest(unittest.TestCase):
    def setUp(self):
        self.out = FakeOut()
        self.d = live.Display(self.out)

    def test_pending_text_comes_before_drafts_in_time_order(self):
        self.d.set_pending("Accurate part.")
        self.d.set_draft(20.0, "second draft")
        self.d.set_draft(10.0, "first draft")
        self.assertEqual(self.out.shown[-1], "Accurate part. first draft second draft")

    def test_a_draft_that_arrives_after_its_accurate_text_is_dropped(self):
        self.d.refined(5.0)
        self.d.set_draft(5.0, "too late")
        self.d.set_draft(4.0, "also too late")
        self.assertEqual(self.d.drafts, {})
        self.d.set_draft(6.0, "next segment")
        self.assertEqual(self.d.drafts, {6.0: "next segment"})

    def test_has_draft_counts_segments_already_replaced(self):
        self.assertFalse(self.d.has_draft(3.0))
        self.d.set_draft(3.0, "hello")
        self.assertTrue(self.d.has_draft(3.0))
        self.d.refined(3.0)
        self.assertTrue(self.d.has_draft(3.0))          # replaced by accurate text: it needs no draft any more
        self.assertFalse(self.d.has_draft(4.0))

    def test_a_draft_is_updated_in_place(self):
        self.d.set_draft(5.0, "so today")
        self.d.set_draft(5.0, "so today we talk about")
        self.assertEqual(self.out.shown[-1], "so today we talk about")

    def test_accurate_result_replaces_its_draft_and_any_older_one(self):
        self.d.set_draft(1.0, "orphan draft of a segment that never got a final")
        self.d.set_draft(5.0, "draft five")
        self.d.set_draft(9.0, "draft nine")
        self.d.refined(5.0)
        self.assertEqual(self.out.shown[-1], "draft nine")

    def test_nothing_left_clears_the_line(self):
        self.d.set_draft(5.0, "draft")
        self.d.refined(5.0)
        self.assertIsNone(self.out.shown[-1])

    def test_clear_forgets_everything(self):
        self.d.set_pending("a")
        self.d.set_draft(1.0, "b")
        self.d.clear()
        self.assertIsNone(self.out.shown[-1])


class SnapshotTest(unittest.TestCase):
    def test_none_until_someone_speaks_then_audio_so_far(self):
        seg = live.Segmenter(max_len=8.0)
        quiet = np.zeros(live.BLOCK, dtype=np.float32)
        loud = (0.1 * np.sin(np.linspace(0, 200, live.BLOCK))).astype(np.float32)
        for _ in range(10):
            seg.feed(quiet)
        self.assertIsNone(seg.snapshot())
        for _ in range(20):
            seg.feed(loud)
        audio, start = seg.snapshot()
        self.assertGreaterEqual(len(audio), 20 * live.BLOCK)
        self.assertGreaterEqual(start, 0.0)


if __name__ == "__main__":
    unittest.main()
