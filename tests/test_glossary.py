import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import glossary  # noqa: E402

RULES = """
# comment line
calcium indicator line = 钙离子指示剂品系   # trailing comment
galvanometric mirror system = 振镜系统
mirror system = 镜子系统
multi-well plate = 多孔板
~ multi-world plate = multi-well plate
~ lava = larvae
"""


def make():
    return glossary.Glossary(RULES.splitlines())


class FixEnglish(unittest.TestCase):
    def test_known_mistakes_are_fixed_including_plural_and_case(self):
        g = make()
        self.assertEqual(g.fix_english("into Multi-World plates."), "into multi-well plates.")
        self.assertEqual(g.fix_english("and lava swimming"), "and larvae swimming")

    def test_whole_words_only(self):
        self.assertEqual(make().fix_english("the lavatory"), "the lavatory")

    def test_hyphen_or_space(self):
        self.assertEqual(make().fix_english("multi world plate"), "multi-well plate")


class Protect(unittest.TestCase):
    def test_longest_term_wins_and_placeholders_are_in_order(self):
        text, tags = make().protect("A galvanometric mirror system and calcium indicator lines.")
        self.assertEqual(text, "A ZorvaxA and ZorvaxB.")
        self.assertEqual(tags["ZorvaxA"], ("galvanometric mirror system", "振镜系统"))
        self.assertEqual(tags["ZorvaxB"], ("calcium indicator lines", "钙离子指示剂品系"))

    def test_shorter_term_still_matches_elsewhere(self):
        text, tags = make().protect("a mirror system, a galvanometric mirror system")
        self.assertEqual(text, "a ZorvaxA, a ZorvaxB")
        self.assertEqual(tags["ZorvaxA"][1], "镜子系统")

    def test_nothing_to_protect(self):
        self.assertEqual(make().protect("Hello there."), ("Hello there.", {}))


class Restore(unittest.TestCase):
    def test_first_use_shows_the_english_then_plain(self):
        g = make()
        _, tags = g.protect("multi-well plates")
        first = g.restore("放入ZorvaxA。", tags)
        self.assertEqual(first, "放入多孔板（multi-well plates）。")
        second = g.restore("再放入ZorvaxA。", tags)
        self.assertEqual(second, "再放入多孔板。")

    def test_a_lost_placeholder_is_reported(self):
        g = make()
        _, tags = g.protect("multi-well plates")
        self.assertIsNone(g.restore("放进去。", tags))

    def test_translator_changing_the_case_or_adding_a_space(self):
        g = make()
        _, tags = g.protect("multi-well plates")
        self.assertIn("多孔板", g.restore("放入zorvax A。", tags))


class Translator(unittest.TestCase):
    def test_wrapper_protects_translates_and_restores(self):
        seen = []

        def base(text):
            seen.append(text)
            return text.replace("Put it in", "放入")

        t = glossary.GlossaryTranslator(base, make())
        self.assertEqual(t("Put it in multi-well plates"), "放入 多孔板（multi-well plates）")
        self.assertEqual(seen, ["Put it in ZorvaxA"])

    def test_falls_back_to_plain_translation_when_a_placeholder_is_lost(self):
        calls = []

        def base(text):
            calls.append(text)
            return "译文"

        t = glossary.GlossaryTranslator(base, make())
        self.assertEqual(t("Put it in multi-well plates"), "译文")
        self.assertEqual(calls, ["Put it in ZorvaxA", "Put it in multi-well plates"])


if __name__ == "__main__":
    unittest.main()
