#!/bin/sh
set -eu
if [ $# -lt 2 ]; then
  echo "Usage: $0 input_image output.svg [potrace_options...]" >&2
  exit 1
fi
input=$1
output=$2
shift 2

cleanup=""
tmp_input="$input"
case "${input##*.}" in
  png|PNG|jpg|JPG|jpeg|JPEG)
    tmp_input="$(mktemp "${TMPDIR:-/tmp}/png_to_svg.XXXXXX.pgm")"
    python3 - "$input" "$tmp_input" <<'PY'
import sys
from PIL import Image

src, dst = sys.argv[1:3]

img = Image.open(src).convert("L")
if img.mode != "1":
    extrema = img.getextrema()
    if extrema != (0, 255):
        img = img.point(lambda p: 255 if p >= 128 else 0, mode="1")
    img = img.convert("L")

width, height = img.size
data = img.tobytes()

with open(dst, "wb") as f:
    f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
    f.write(data)
PY
    cleanup="$tmp_input"
    ;;
esac

potrace "$tmp_input" --svg -o "$output" "$@"

if [ -n "$cleanup" ]; then
  rm -f "$cleanup"
fi
