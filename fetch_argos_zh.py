#!/usr/bin/env python3
"""Download the offline English->Chinese translation model (Argos Translate, ~100 MB) and unpack only the parts we use.

Why not `pip install argostranslate`? It pulls in PyTorch (~1 GB); we only need the CTranslate2 model and the
SentencePiece vocabulary, which live-captions loads directly.
"""
import io
import json
import os
import shutil
import ssl
import sys
import tempfile
import urllib.request
import zipfile

import certifi

INDEX = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
DEST = os.path.expanduser(os.environ.get("ARGOS_ZH_DIR") or "~/.local/share/argos-translate/packages/translate-en_zh-1_9")


def get(url):
    ctx = ssl.create_default_context(cafile=certifi.where())
    # the model server answers 403 to urllib's default "Python-urllib" agent, so identify ourselves honestly
    req = urllib.request.Request(url, headers={"User-Agent": "lecture-captions/1.0 (+https://github.com/SyzenShen/lecture-captions)"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
        return r.read()


def main():
    if os.path.exists(os.path.join(DEST, "model", "model.bin")):
        print(f"already installed: {DEST}")
        return
    print("looking up the en->zh package ...")
    pk = next(p for p in json.loads(get(INDEX)) if p["from_code"] == "en" and p["to_code"] == "zh")
    print(f"downloading {pk['links'][0]} ...")
    data = get(pk["links"][0])
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
        src = next(os.path.dirname(os.path.dirname(os.path.join(r, f))) for r, _, fs in os.walk(tmp) for f in fs
                   if f == "model.bin")
        os.makedirs(DEST, exist_ok=True)
        shutil.copytree(os.path.join(src, "model"), os.path.join(DEST, "model"), dirs_exist_ok=True)
        shutil.copy(os.path.join(src, "sentencepiece.model"), DEST)
        for extra in ("metadata.json", "README.md"):
            if os.path.exists(os.path.join(src, extra)):
                shutil.copy(os.path.join(src, extra), DEST)
    print(f"installed to {DEST}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"could not install the translation model: {e}\nYou can still run with --translator none (English only).")
