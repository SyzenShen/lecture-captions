import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import refine  # noqa: E402

START = datetime(2026, 10, 23, 10, 51, 0)


class SessionStart(unittest.TestCase):
    def test_from_file_name(self):
        self.assertEqual(refine.session_start("transcripts/2026-10-23_1051.wav"), START.replace(second=0))

    def test_unknown_name_falls_back_to_now(self):
        self.assertIsInstance(refine.session_start("lecture.mp3"), datetime)


class EntriesFromSegments(unittest.TestCase):
    def test_one_entry_per_sentence_with_clock_times(self):
        entries = refine.entries_from_segments([(60.0, 70.0, "First sentence here. Second one is a question?")], START)
        self.assertEqual([e["en"] for e in entries], ["First sentence here.", "Second one is a question?"])
        self.assertEqual(entries[0]["ts"], "10:52:00")
        self.assertGreater(entries[1]["ts"], entries[0]["ts"])
        self.assertEqual(entries[0]["q"], "")
        self.assertEqual(entries[1]["q"], "Second one is a question?")

    def test_garbage_is_dropped(self):
        segs = [(0, 5, "你好"), (5, 9, "ooooohhhhhhhhhhhhhhhhhh"), (9, 12, "Thank you."), (12, 15, "A real sentence about proteins.")]
        self.assertEqual([e["en"] for e in refine.entries_from_segments(segs, START)], ["A real sentence about proteins."])

    def test_empty_segments_give_no_entries(self):
        self.assertEqual(refine.entries_from_segments([(0, 1, "   ")], START), [])


if __name__ == "__main__":
    unittest.main()
