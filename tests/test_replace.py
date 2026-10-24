import contextlib
import io
import os
import queue
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import live  # noqa: E402


class SpreadTimes(unittest.TestCase):
    def test_even_spread(self):
        self.assertEqual(live.spread_times("10:00:00", "10:00:30", 4), ["10:00:00", "10:00:10", "10:00:20", "10:00:30"])

    def test_one_sentence_takes_the_first_time(self):
        self.assertEqual(live.spread_times("10:00:05", "10:00:30", 1), ["10:00:05"])

    def test_never_runs_backwards(self):
        self.assertEqual(live.spread_times("10:00:30", "10:00:10", 2), ["10:00:30", "10:00:30"])


class ReplaceBlock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = live.Output(os.path.join(self.tmp.name, "t.txt"))
        self.quiet = io.StringIO()
        self.events = queue.Queue()
        self.out.subs.append(self.events)
        self.ids = {}
        for name, en, block in [("a1", "Alpha one.", 0), ("a2", "Alpha two?", 0), ("b1", "Beta one.", 1), ("b2", "Beta two.", 1), ("c1", "Gamma.", 2)]:
            with contextlib.redirect_stdout(self.quiet):
                self.ids[name] = self.out.emit(en, block=block)
        while not self.events.empty():
            self.events.get()

    def tearDown(self):
        self.out.f.close()
        self.tmp.cleanup()

    def texts(self):
        return [e["en"] for e in self.out.entries]

    def test_the_block_is_replaced_in_place_and_order_is_kept(self):
        new = self.out.replace_block(1, ["Better beta one.", "Better beta two.", "Better beta three."])
        self.assertEqual(self.texts(), ["Alpha one.", "Alpha two?", "Better beta one.", "Better beta two.", "Better beta three.", "Gamma."])
        self.assertEqual(len(new), 3)

    def test_ids_stay_unique_and_old_ones_are_gone(self):
        new = self.out.replace_block(0, ["Only one sentence."])
        ids = [e["id"] for e in self.out.entries]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn(self.ids["a1"], self.out.by_id)
        self.assertIn(new[0][0], self.out.by_id)

    def test_the_web_page_is_told_what_to_drop_and_what_to_insert(self):
        self.out.replace_block(1, ["Better."])
        ev = self.events.get_nowait()
        self.assertEqual(ev["type"], "replace")
        self.assertEqual(ev["drop"], [self.ids["b1"], self.ids["b2"]])
        self.assertEqual([e["en"] for e in ev["entries"]], ["Better."])

    def test_a_translation_for_a_replaced_sentence_is_ignored(self):
        self.out.replace_block(1, ["Better."])
        self.out.set_zh(self.ids["b1"], "旧译文")
        self.assertNotIn("旧译文", [e["zh"] for e in self.out.entries])

    def test_questions_are_recomputed(self):
        self.out.replace_block(0, ["Why does the cell do that?"])
        self.assertEqual([e["q"] for e in self.out.entries][0], "Why does the cell do that?")

    def test_unknown_block_or_empty_result_changes_nothing(self):
        before = self.texts()
        self.assertEqual(self.out.replace_block(99, ["x."]), [])
        self.assertEqual(self.out.replace_block(1, []), [])
        self.assertEqual(self.texts(), before)

    def test_a_result_that_lost_most_of_the_speech_is_not_accepted(self):
        before = self.texts()
        self.assertEqual(self.out.replace_block(1, ["Beta."], min_ratio=0.6), [])      # 1 word for 4
        self.assertEqual(self.texts(), before)
        self.assertEqual(len(self.out.replace_block(1, ["Beta one two three."], min_ratio=0.6)), 1)

    def test_notes_use_the_refined_text(self):
        self.out.replace_block(1, ["Better beta."])
        md = live.render_markdown(self.out.entries, self.out.started)
        self.assertIn("Better beta.", md)
        self.assertNotIn("Beta one.", md)


if __name__ == "__main__":
    unittest.main()
