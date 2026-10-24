import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import live  # noqa: E402


def chunk(start, end, forced=False):
    return (b"audio", start, end, forced)


class BlockGrouperTests(unittest.TestCase):
    def test_off_never_ends_a_block(self):
        g = live.BlockGrouper(0)
        for i in range(5):
            self.assertEqual(g.tag(chunk(i * 6, i * 6 + 6))[4:], (0, False))

    def test_block_ends_once_enough_speech_has_gathered(self):
        g = live.BlockGrouper(15)
        tags = [g.tag(chunk(i * 6, i * 6 + 6))[4:] for i in range(6)]
        # 6 s, 12 s, 18 s (ends block 0), then block 1 starts again
        self.assertEqual(tags, [(0, False), (0, False), (0, True), (1, False), (1, False), (1, True)])

    def test_close_starts_a_new_block(self):
        g = live.BlockGrouper(15)
        self.assertEqual(g.tag(chunk(0, 6))[4:], (0, False))
        g.close()
        self.assertEqual(g.tag(chunk(100, 106))[4:], (1, False))

    def test_close_without_open_block_changes_nothing(self):
        g = live.BlockGrouper(15)
        g.close()
        self.assertEqual(g.tag(chunk(0, 6))[4], 0)

    def test_audio_and_times_pass_through(self):
        g = live.BlockGrouper(15)
        self.assertEqual(g.tag(chunk(3, 9, True))[:4], (b"audio", 3, 9, True))


if __name__ == "__main__":
    unittest.main()
