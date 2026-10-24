import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cloud_eval  # noqa: E402

SEEN = []


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        SEEN.append({"path": self.path, "auth": self.headers.get("Authorization"), "ctype": self.headers.get("Content-Type"), "body": body})
        if "/listen" in self.path:
            out = {"results": {"channels": [{"alternatives": [{"transcript": "deepgram text"}]}]}}
        else:
            out = {"text": "openai style text"}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class CloudEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        os.environ["CLOUD_EVAL_BASE_URL"] = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        os.environ.pop("CLOUD_EVAL_BASE_URL", None)

    def setUp(self):
        SEEN.clear()

    def test_openai_style_request_and_answer(self):
        text = cloud_eval.transcribe_chunk(cloud_eval.PROVIDERS["groq"], "KEY", b"RIFFdata", "Biology: gene, protein", [])
        self.assertEqual(text, "openai style text")
        req = SEEN[0]
        self.assertEqual(req["path"], "/openai/v1/audio/transcriptions")
        self.assertEqual(req["auth"], "Bearer KEY")
        self.assertTrue(req["ctype"].startswith("multipart/form-data; boundary="))
        for needle in (b'name="model"', b"whisper-large-v3-turbo", b'name="language"', b'name="prompt"', b"Biology: gene, protein", b"RIFFdata"):
            self.assertIn(needle, req["body"])

    def test_deepgram_request_sends_keyterms_and_reads_transcript(self):
        text = cloud_eval.transcribe_chunk(cloud_eval.PROVIDERS["deepgram"], "KEY", b"RIFFdata", "", ["helicase", "primer"])
        self.assertEqual(text, "deepgram text")
        req = SEEN[0]
        self.assertEqual(req["auth"], "Token KEY")
        self.assertIn("keyterm=helicase", req["path"])
        self.assertIn("keyterm=primer", req["path"])
        self.assertEqual(req["body"], b"RIFFdata")

    def test_nothing_is_sent_without_yes(self):
        self.assertEqual(len(SEEN), 0)

    def test_distance(self):
        self.assertEqual(cloud_eval.distance("the cat sat", "the cat sat"), 0)
        self.assertAlmostEqual(cloud_eval.distance("the cat sat", "the dog sat"), 1 / 3)
        self.assertAlmostEqual(cloud_eval.distance("A-B c", "a b c"), 0)


if __name__ == "__main__":
    unittest.main()
