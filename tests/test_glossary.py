import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import glossary  # noqa: E402

P = glossary.strip_marks

RULES = """
# comment line
calcium indicator line = 钙离子指示剂品系   # trailing comment
galvanometric mirror system = 振镜系统
mirror system = 镜子系统
multi-well plate = 多孔板
newbie tracker = -
signal = 信号 !
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
        self.assertEqual(text, "A ZorvaxA and ZorvaxB .")
        self.assertEqual(tags["ZorvaxA"], ("galvanometric mirror system", "振镜系统", False))
        self.assertEqual(tags["ZorvaxB"], ("calcium indicator lines", "钙离子指示剂品系", False))

    def test_shorter_term_still_matches_elsewhere(self):
        text, tags = make().protect("a mirror system, a galvanometric mirror system")
        self.assertEqual(text, "a ZorvaxA , a ZorvaxB")
        self.assertEqual(tags["ZorvaxA"][1], "镜子系统")

    def test_nothing_to_protect(self):
        self.assertEqual(make().protect("Hello there."), ("Hello there.", {}))


class Restore(unittest.TestCase):
    def test_first_use_shows_the_english_then_plain(self):
        g = make()
        _, tags = g.protect("multi-well plates")
        first = g.restore("放入ZorvaxA。", tags)
        self.assertEqual(P(first), "放入多孔板（multi-well plates）。")
        second = g.restore("再放入ZorvaxA。", tags)
        self.assertEqual(P(second), "再放入多孔板。")

    def test_a_term_without_a_translation_is_shown_as_its_english_in_brackets(self):
        g = make()
        _, tags = g.protect("a newbie tracker here")
        self.assertEqual(P(g.restore("这里有ZorvaxA。", tags)), "这里有（newbie tracker）。")
        self.assertEqual(P(g.restore("这里有ZorvaxA。", tags)), "这里有（newbie tracker）。")     # every time

    def test_bang_rule_and_the_always_option_bracket_every_time(self):
        g = make()
        _, tags = g.protect("the signal")
        self.assertEqual(P(g.restore("ZorvaxA强。", tags)), "信号（signal）强。")
        self.assertEqual(P(g.restore("ZorvaxA强。", tags)), "信号（signal）强。")
        g2 = glossary.Glossary(RULES.splitlines(), always=True)
        _, tags = g2.protect("multi-well plates")
        g2.restore("ZorvaxA", tags)
        self.assertEqual(P(g2.restore("ZorvaxA", tags)), "多孔板（multi-well plates）")

    def test_terms_are_wrapped_in_markers_for_the_web_page(self):
        g = make()
        _, tags = g.protect("multi-well plates")
        out = g.restore("放入ZorvaxA。", tags)
        self.assertEqual(out, "放入" + glossary.MARK_S + "多孔板（multi-well plates）" + glossary.MARK_E + "。")
        self.assertEqual(P(out), "放入多孔板（multi-well plates）。")

    def test_no_stray_space_before_chinese_punctuation(self):
        g = make()
        _, tags = g.protect("multi-well plates.")
        self.assertEqual(P(g.restore("放入ZorvaxA 。", tags)), "放入多孔板（multi-well plates）。")

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
        self.assertEqual(P(t("Put it in multi-well plates")), "放入 多孔板（multi-well plates）")
        self.assertEqual(seen, ["Put it in ZorvaxA"])

    def test_a_sentence_that_is_only_the_term_is_not_sent_to_the_translator(self):
        def base(text):
            raise AssertionError("must not be called")

        t = glossary.GlossaryTranslator(base, make())
        self.assertEqual(P(t("Multi-well plates.")), "多孔板（multi-well plates）。")

    def test_falls_back_to_plain_translation_when_a_placeholder_is_lost(self):
        calls = []

        def base(text):
            calls.append(text)
            return "译文"

        t = glossary.GlossaryTranslator(base, make())
        self.assertEqual(P(t("Put it in multi-well plates")), "译文")
        self.assertEqual(calls, ["Put it in ZorvaxA", "Put it in multi-well plates"])


class TranslationGuards(unittest.TestCase):
    def test_a_repeated_phrase_is_squashed(self):
        import live
        self.assertEqual(live.clean_zh("肩相接相接相接相接相接相接", "Shoulder to back."), "肩相接")
        self.assertEqual(live.clean_zh("这是测试，这是测试，这是测试，结束", "x" * 30), "这是测试，结束")

    def test_normal_repetition_is_left_alone(self):
        import live
        self.assertEqual(live.clean_zh("我们可以看到，看到的结果。", "We can see the result."), "我们可以看到，看到的结果。")

    def test_interjections_are_not_translated(self):
        import live
        t = live.ArgosTranslator.__new__(live.ArgosTranslator)      # no model needed: it returns before using it
        self.assertEqual(P(t("Ooh!")), "")
        self.assertEqual(P(t("Hmm.")), "")


if __name__ == "__main__":
    unittest.main()
