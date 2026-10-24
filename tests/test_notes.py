import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import live  # noqa: E402

STARTED = datetime(2026, 10, 23, 10, 51)
SAVED = datetime(2026, 10, 23, 12, 0, 5)


def entry(ts, en, zh="", q=""):
    return {"ts": ts, "en": en, "zh": zh, "q": q}


class RenderMarkdown(unittest.TestCase):
    def test_header_and_entries(self):
        md = live.render_markdown([entry("10:51:37", "Hello.", "你好。")], STARTED, "Biology lecture", SAVED)
        self.assertIn("# 课堂记录 2026-10-23 10:51", md)
        self.assertIn("> 主题提示：Biology lecture", md)
        self.assertIn("共 1 句 · 保存于 12:00:05", md)
        self.assertIn("**10:51:37** Hello.  \n你好。", md)

    def test_untranslated_sentence_has_no_empty_line_for_chinese(self):
        md = live.render_markdown([entry("10:51:37", "Hello."), entry("10:51:40", "Next.", "下一句。")], STARTED, saved_at=SAVED)
        lines = md.split("\n")
        i = lines.index("**10:51:37** Hello.  ")
        self.assertEqual(lines[i + 1], "")                              # nothing between the English and the next sentence
        self.assertEqual(lines[i + 2], "**10:51:40** Next.  ")
        self.assertNotIn("主题提示", md)

    def test_question_list_is_appended_only_when_there_are_questions(self):
        none = live.render_markdown([entry("10:51:37", "Hello.")], STARTED, saved_at=SAVED)
        self.assertNotIn("课堂提问", none)
        some = live.render_markdown([entry("10:52:00", "Why?", q="Why is it so?")], STARTED, saved_at=SAVED)
        self.assertIn("## 课堂提问（1）", some)
        self.assertIn("- **10:52:00** Why is it so?", some)


if __name__ == "__main__":
    unittest.main()
