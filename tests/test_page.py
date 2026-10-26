import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import live  # noqa: E402


def found(text):
    return [m.group(0) for m in re.finditer(live.connective_pattern(), text, re.I)]


class Connectives(unittest.TestCase):
    def test_logical_words_are_found_in_any_case(self):
        self.assertEqual(found("But because of this, so we use it. If not, then no. For example, this."),
                         ["But", "because", "so", "If", "then", "For example"])

    def test_longest_phrase_wins(self):
        self.assertEqual(found("As a result we stop. On the other hand, we go."), ["As a result", "On the other hand"])

    def test_whole_words_only(self):
        self.assertEqual(found("the sofa, a thought, a wife, soft"), [])

    def test_and_or_are_not_marked(self):
        self.assertEqual(found("one and two or three"), [])

    def test_abbreviations(self):
        self.assertEqual(found("e.g. this, i.e. that"), ["e.g.", "i.e."])

    def test_pattern_goes_into_the_page_as_a_valid_javascript_string(self):
        page = live.PAGE.replace("__CONN__", json.dumps(live.connective_pattern()))
        self.assertNotIn("__CONN__", page)
        self.assertIn('new RegExp("(?<![\\\\w.])', page)


class ChineseConnectives(unittest.TestCase):
    def found(self, text):
        return [m.group(0) for m in re.finditer(live.connective_pattern_zh(), text)]

    def test_clear_connectives_are_found_anywhere(self):
        self.assertEqual(self.found("这很方便，因为我们可以放入，所以很快。如果你记得，但是不对"), ["因为", "所以", "如果", "但是"])

    def test_a_single_character_but_only_starts_a_clause(self):
        self.assertEqual(self.found("但我们可以。很但愿如此"), ["但"])

    def test_clause_words_must_start_a_clause(self):
        self.assertEqual(self.found("然后我们看，最后一个。并不然后"), ["然后", "最后"])

    def test_plain_text_has_none(self):
        self.assertEqual(self.found("蛋白质功能会容忍类似酸中物理化学物质的交换"), [])

    def test_pattern_is_injected_into_the_page(self):
        page = live.PAGE.replace("__CONNZH__", json.dumps(live.connective_pattern_zh()))
        self.assertNotIn("__CONNZH__", page)


if __name__ == "__main__":
    unittest.main()
