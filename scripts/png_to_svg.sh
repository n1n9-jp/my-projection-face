#!/bin/sh
set -eu
if [ $# -lt 2 ]; then
  echo "Usage: $0 input.png output.svg [potrace_options...]" >&2
  exit 1
fi
input=$1
output=$2
shift 2
potrace "$input" --svg -o "$output" "$@"
