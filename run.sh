#!/bin/bash
cd "$(dirname "$0")"
[ -f config.local.sh ] && . ./config.local.sh        # optional, git-ignored: e.g. export CAPTIONS_NOTES_DIR=~/Documents/notes
export SSL_CERT_FILE="$(.venv/bin/python -c 'import certifi;print(certifi.where())')"
exec .venv/bin/python live.py "$@"
