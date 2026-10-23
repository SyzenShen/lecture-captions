import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from questions import extract_question, is_question  # noqa: E402


class IsQuestion(unittest.TestCase):
    def check(self, cases):
        for sentence, expected in cases:
            with self.subTest(sentence=sentence):
                self.assertEqual(is_question(sentence), expected)

    def test_real_questions(self):
        self.check([
            ("Who knows about the gene numbers in humans?", True),
            ("So how big is the dose?", True),
            ("and what's this called?", True),
            ("A mass spectroscopy?", True),
            ("Where and when?", True),
        ])

    def test_word_order_without_question_mark(self):
        self.check([
            ("Do you know the answer", True),
            ("Is that clear.", True),
            ("Are there any questions", True),
            ("How many pieces of information do we have in each position", True),
            ("What about the other strand", True),
        ])

    def test_filler_is_not_a_question(self):
        self.check([
            ("and so on, right?", False),
            ("It works like this, you know?", False),
            ("Yeah?", False),
            ("So why?", False),
            ("Why?", False),
        ])

    def test_statements_are_not_questions(self):
        self.check([
            ("Do not forget the exam.", False),
            ("Don't worry about it.", False),
            ("What we see here is a protein.", False),
            ("How it works is complicated.", False),
            ("who is also kind of crazy about it.", False),
            ("We use proteins for DNA replication.", False),
            ("", False),
        ])


class ExtractQuestion(unittest.TestCase):
    def test_picks_the_question_out_of_several_sentences(self):
        self.assertEqual(extract_question("That is the idea. So how does the cell know? We will see."), "So how does the cell know?")

    def test_no_question(self):
        self.assertEqual(extract_question("This is just a statement. Nothing else."), "")

    def test_long_question_is_shortened_to_its_end(self):
        text = " ".join(["word"] * 40) + " which one was it?"
        shown = extract_question(text)
        self.assertTrue(shown.startswith("… "))
        self.assertTrue(shown.endswith("which one was it?"))
        self.assertEqual(len(shown.split()), 25)  # "…" plus 24 words


if __name__ == "__main__":
    unittest.main()
