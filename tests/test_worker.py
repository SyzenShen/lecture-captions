import contextlib
import io
import os
import queue
import sys
import tempfile
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import live  # noqa: E402


class FakeSeg:
    speaking, speech_start, t = False, None, 0.0


class FakeDisplay:
    def __init__(self):
        self.refined_upto = []

    def refined(self, upto):
        self.refined_upto.append(upto)

    def set_pending(self, text):
        pass

    def clear(self):
        pass


def chunk(start, seconds, forced=False, block=0, ends=False, decode_now=False):
    return (np.zeros(int(seconds * live.SR), dtype=np.float32), start, start + seconds, forced, block, ends, decode_now)


class MergingWorker(unittest.TestCase):
    def run_worker(self, items, seg_t=100.0):
        self.calls = []

        def fake_transcribe(model, audio, context, beam=None, careful=False, as_segments=False):
            self.calls.append(len(audio) / live.SR)
            return f"Sentence {len(self.calls)}."

        real = live.transcribe
        live.transcribe = fake_transcribe
        tmp = tempfile.TemporaryDirectory()
        out = live.Output(os.path.join(tmp.name, "t.txt"))
        seg, display, q, tq = FakeSeg(), FakeDisplay(), queue.Queue(), queue.Queue()
        seg.t = seg_t
        for it in items:
            q.put(it)
        q.put(None)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                t = threading.Thread(target=live.worker, args=(q, None, tq, out, seg, display))
                t.start()
                t.join(10)
        finally:
            live.transcribe = real
            out.f.close()
        self.out, self.display = out, display
        tmp.cleanup()

    def test_pieces_are_decoded_together_when_the_group_ends(self):
        self.run_worker([chunk(0, 3), chunk(3.5, 4), chunk(8, 3, decode_now=True)])
        self.assertEqual(len(self.calls), 1)
        self.assertAlmostEqual(self.calls[0], 10.0, places=2)      # the audio of all three pieces
        self.assertEqual(self.display.refined_upto, [8.0])          # drafts up to the last piece are replaced

    def test_nothing_is_lost_when_the_lecture_ends_inside_a_group(self):
        self.run_worker([chunk(0, 3), chunk(3.5, 4)])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([e["en"] for e in self.out.entries], ["Sentence 1."])

    def test_without_merging_every_piece_is_decoded(self):
        self.run_worker([chunk(0, 3, decode_now=True), chunk(5, 4, decode_now=True)])
        self.assertEqual(len(self.calls), 2)

    def test_pause_flush_decodes_waiting_pieces(self):
        self.run_worker([chunk(0, 3), "flush", chunk(30, 3, decode_now=True)])
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()
