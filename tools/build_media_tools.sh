#!/usr/bin/env bash
# Build the fixed decoder CLI; no GPU, network or external codec libraries.
set -euo pipefail

SCRIPT="$(realpath -- "$0")"
ARCHIVE="$(realpath -- "$1")"
OUTPUT="$(realpath -m -- "$2")"
EXPECTED=ab4652ccd1ca1a5c864c09751bb0465757f8a4aafc00425b7490e3bdd9b6fccf
printf '%s  %s\n' "$EXPECTED" "$ARCHIVE" | sha256sum --check --status
mkdir -- "$OUTPUT"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/deepfake-ffmpeg-build.XXXXXX")"
trap 'rm -rf -- "$WORK"' EXIT
tar -xf "$ARCHIVE" -C "$WORK" --strip-components=1
cd "$WORK"
printf '%s\n' N-127271-gba987fe24d-local-lgpl > FF_VERSION
OPTIONS=(
  --disable-autodetect --disable-gpl --disable-nonfree --disable-version3
  --disable-shared --enable-static --disable-doc --disable-debug
  --disable-ffplay --disable-avdevice --disable-network --disable-hwaccels
  --disable-decoder=av1 --disable-encoders
  --enable-encoder=rawvideo,pcm_f32le,pcm_s16le --disable-muxers
  --enable-muxer=rawvideo,pcm_f32le,pcm_s16le,wav --disable-protocols
  --enable-protocol=file,pipe
)
./configure "${OPTIONS[@]}"
make -j "${DEEPFAKE_BUILD_JOBS:-4}" ffmpeg ffprobe
mkdir -p "$OUTPUT/bin" "$OUTPUT/source" "$OUTPUT/licenses"
cp -- ffmpeg ffprobe "$OUTPUT/bin/"
cp -- "$ARCHIVE" "$OUTPUT/source/ffmpeg-ba987fe24d.tar.gz"
cp -- LICENSE.md COPYING.LGPLv2.1 "$OUTPUT/licenses/"
cp -- "$SCRIPT" "$OUTPUT/build.sh"
python3 - "$OUTPUT" "${OPTIONS[@]}" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
samples = bytes(64)
decoded = subprocess.run([
    str(root / 'bin/ffmpeg'), '-v', 'error', '-f', 'f32le', '-ar', '16000',
    '-ac', '1', '-i', 'pipe:0', '-af', 'pan=mono|c0=c0', '-f', 'f32le',
    '-acodec', 'pcm_f32le', 'pipe:1',
], input=samples, capture_output=True, check=True).stdout
assert decoded == samples, 'fixed float-audio CLI smoke failed'
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
tools = {}
for name in ('ffmpeg', 'ffprobe'):
    path = root / 'bin' / name
    tools[name] = {
        'sha256': sha(path), 'bytes': path.stat().st_size,
        'version': subprocess.check_output([str(path), '-version'], text=True),
        'dynamic_libraries': subprocess.check_output(['ldd', str(path)], text=True),
    }
source = root / 'source/ffmpeg-ba987fe24d.tar.gz'
metadata = {
    'schema': 1, 'version': 'N-127271-gba987fe24d-local-lgpl', 'tools': tools,
    'source': {
        'archive': str(source.relative_to(root)), 'sha256': sha(source),
        'url': 'https://github.com/FFmpeg/FFmpeg/archive/ba987fe24d6713c4d4b3a15cebaeca18c41fbd31.tar.gz',
        'commit': 'ba987fe24d6713c4d4b3a15cebaeca18c41fbd31',
        'verification': 'HTTPS archive, fixed SHA256 and Git archive PAX commit',
    },
    'source_modifications': ['FF_VERSION labels the pinned Git source and local build'],
    'configure': sys.argv[2:], 'compiler': subprocess.check_output(
        ['cc', '--version'], text=True).splitlines()[0],
    'platform': platform.platform(), 'license': 'LGPL-2.1-or-later',
    'limitations': [
        'No AV1 software decoder or external codec libraries in this build.',
        'No network protocols, hardware decoding or libx264 encoder.',
        'FFmpeg libraries are static; host glibc and libm remain required.',
        'Build-time demonstrations use the builder host FFmpeg.',
    ],
}
(root / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
PY
