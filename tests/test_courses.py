import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import courses  # noqa: E402


class Courses(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prompts = os.path.join(self.tmp.name, "prompts")
        self.gloss = os.path.join(self.tmp.name, "glossary")
        os.makedirs(self.prompts)
        os.makedirs(self.gloss)
        for name in ["ngs_full", "ngs_0_linux", "ngs_3_rnaseq", "ngs_simple", "molbio_pcr_editing", "estimation_short", "synbio_1_cell"]:
            open(os.path.join(self.prompts, name + ".txt"), "w").write("x")
        for name in ["ngs", "neuro_imaging"]:
            open(os.path.join(self.gloss, name + ".txt"), "w").write("x")
        self.c = courses.scan(self.prompts, self.gloss)

    def tearDown(self):
        self.tmp.cleanup()

    def test_files_are_grouped_by_their_first_word(self):
        self.assertEqual(sorted(self.c), ["estimation", "molbio", "neuro", "ngs", "synbio"])
        self.assertEqual(sorted(self.c["ngs"]["prompts"]), ["0", "3", "full", "simple"])
        self.assertEqual(len(self.c["ngs"]["glossaries"]), 1)

    def test_glossary_only_course(self):
        self.assertEqual(self.c["neuro"]["prompts"], {})
        self.assertIsNone(courses.pick_prompt(self.c["neuro"]))

    def test_name_may_be_a_unique_prefix(self):
        self.assertEqual(courses.resolve("mol", self.c)[0], "molbio")
        self.assertEqual(courses.resolve("NGS", self.c)[0], "ngs")
        self.assertIsNone(courses.resolve("zzz", self.c)[0])

    def test_ambiguous_prefix_lists_candidates(self):
        key, matches = courses.resolve("n", self.c)
        self.assertIsNone(key)
        self.assertEqual(matches, ["neuro", "ngs"])

    def test_without_a_lesson_the_full_prompt_wins(self):
        self.assertTrue(courses.pick_prompt(self.c["ngs"]).endswith("ngs_full.txt"))

    def test_a_single_file_or_a_short_one_is_used_when_there_is_no_full(self):
        self.assertTrue(courses.pick_prompt(self.c["molbio"]).endswith("molbio_pcr_editing.txt"))
        self.assertTrue(courses.pick_prompt(self.c["estimation"]).endswith("estimation_short.txt"))

    def test_only_lessons_falls_back_to_the_first_lesson(self):
        self.assertTrue(courses.pick_prompt(self.c["synbio"]).endswith("synbio_1_cell.txt"))

    def test_lesson_selection(self):
        self.assertTrue(courses.pick_prompt(self.c["ngs"], 3).endswith("ngs_3_rnaseq.txt"))
        self.assertIsNone(courses.pick_prompt(self.c["ngs"], 7))
        self.assertEqual(courses.lessons(self.c["ngs"]), ["0", "3"])

    def test_last_course_is_remembered(self):
        path = os.path.join(self.tmp.name, ".last")
        self.assertIsNone(courses.read_last(path))
        courses.write_last(path, "ngs")
        self.assertEqual(courses.read_last(path), "ngs")


class PickCourse(unittest.TestCase):
    """The interactive part in live.py: ask, remember, fill in the prompt file and glossaries."""

    def setUp(self):
        import argparse
        import builtins
        import contextlib
        import io
        import live
        self.live, self.argparse, self.builtins, self.contextlib, self.io = live, argparse, builtins, contextlib, io
        self.tmp = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.tmp.name, "prompts"))
        os.makedirs(os.path.join(self.tmp.name, "glossary"))
        for name in ["ngs_full", "ngs_3_rnaseq", "synbio_full"]:
            open(os.path.join(self.tmp.name, "prompts", name + ".txt"), "w").write("x")
        open(os.path.join(self.tmp.name, "glossary", "ngs.txt"), "w").write("x")
        self.all = courses.scan(os.path.join(self.tmp.name, "prompts"), os.path.join(self.tmp.name, "glossary"))
        self.last_path = os.path.join(self.tmp.name, ".last")
        self.saved = live.LAST_COURSE
        live.LAST_COURSE = self.last_path

    def tearDown(self):
        self.live.LAST_COURSE = self.saved
        self.tmp.cleanup()

    def args(self, course=None, lesson=None):
        return self.argparse.Namespace(course=course, lesson=lesson, prompt_file=None, glossary=[])

    def ask(self, answer, a):
        real_input, real_isatty = self.builtins.input, self.live.sys.stdin.isatty
        self.builtins.input = lambda prompt="": answer
        self.live.sys.stdin.isatty = lambda: True
        try:
            with self.contextlib.redirect_stdout(self.io.StringIO()):
                return self.live.pick_course(a, self.all)
        finally:
            self.builtins.input, self.live.sys.stdin.isatty = real_input, real_isatty

    def test_name_on_the_command_line_sets_prompt_glossary_and_is_remembered(self):
        a = self.args("ngs", "3")
        self.assertEqual(self.live.pick_course(a, self.all), "ngs 第3课")
        self.assertTrue(a.prompt_file.endswith("ngs_3_rnaseq.txt"))
        self.assertEqual(len(a.glossary), 1)
        self.assertEqual(courses.read_last(self.last_path), "ngs")

    def test_enter_alone_repeats_the_last_course(self):
        courses.write_last(self.last_path, "synbio")
        a = self.args()
        self.assertEqual(self.ask("", a), "synbio")
        self.assertTrue(a.prompt_file.endswith("synbio_full.txt"))

    def test_a_number_picks_from_the_sorted_list(self):
        a = self.args()
        self.assertEqual(self.ask("2", a), "synbio")          # ngs, synbio

    def test_name_and_lesson_can_be_typed_together(self):
        a = self.args()
        self.assertEqual(self.ask("ngs 3", a), "ngs 第3课")

    def test_without_a_terminal_and_without_a_name_there_is_no_course(self):
        courses.write_last(self.last_path, "ngs")
        self.assertIsNone(self.live.pick_course(self.args(), self.all))

    def test_unknown_course_exits_with_a_message(self):
        with self.assertRaises(SystemExit):
            self.live.pick_course(self.args("zzz"), self.all)


if __name__ == "__main__":
    unittest.main()
