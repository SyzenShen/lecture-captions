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

    def test_waits_for_a_real_pause_once_long_enough(self):
        g = live.BlockGrouper(15)
        tags = [g.tag(chunk(i * 6, i * 6 + 6, forced=i < 3))[4:] for i in range(5)]
        # forced splits (no pause) do not end the block at 18 s; the first real pause after 15 s does
        self.assertEqual(tags, [(0, False), (0, False), (0, False), (0, True), (1, False)])

    def test_hard_cap_even_without_a_pause(self):
        g = live.BlockGrouper(10)
        tags = [g.tag(chunk(i * 6, i * 6 + 6, forced=True))[4:] for i in range(4)]
        # 6, 12, 18 s (1.8 x 10 s: ends), then a new block
        self.assertEqual(tags, [(0, False), (0, False), (0, True), (1, False)])

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


class BlockAudioTests(unittest.TestCase):
    def test_finished_block_goes_to_the_refiner_queue_with_all_its_audio(self):
        import numpy as np
        import queue
        rq = queue.Queue()
        ba = live.BlockAudio(rq)
        ba.add((np.ones(3), 0, 1, False, 0, False))
        self.assertTrue(rq.empty())
        ba.add((np.zeros(2), 1, 2, False, 0, True))
        block, audio = rq.get_nowait()
        self.assertEqual((block, len(audio)), (0, 5))
        ba.close()                       # nothing pending: nothing sent
        self.assertTrue(rq.empty())

    def test_close_sends_an_unfinished_block(self):
        import numpy as np
        import queue
        rq = queue.Queue()
        ba = live.BlockAudio(rq)
        ba.add((np.ones(4), 0, 1, False, 7, False))
        ba.close()
        self.assertEqual(rq.get_nowait()[0], 7)


class RefinedSentences(unittest.TestCase):
    def test_punctuated_text_is_split_at_sentence_ends(self):
        self.assertEqual(live.refined_sentences(["Hello there. How are you?", "Fine."]), ["Hello there.", "How are you?", "Fine."])

    def test_a_sentence_split_over_two_segments_is_joined(self):
        self.assertEqual(live.refined_sentences(["The structure of", "the protein is stable."]), ["The structure of the protein is stable."])

    def test_unpunctuated_run_on_is_broken_up(self):
        words = " ".join(f"w{i}" for i in range(70))
        sents = live.refined_sentences([words])
        self.assertTrue(all(len(x.split()) <= 30 for x in sents))
        self.assertEqual(" ".join(sents), words)

    def test_empty(self):
        self.assertEqual(live.refined_sentences([]), [])


if __name__ == "__main__":
    unittest.main()
