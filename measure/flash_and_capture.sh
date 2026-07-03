#!/usr/bin/env bash
# Compile, flash and capture one model sketch on ESP32 or Pico.
# Run on the machine with the board attached (needs arduino-cli + cores +
# TFLite Micro libs installed, plus pyserial for capture_serial.py).
#
# Usage:
#   ./flash_and_capture.sh <model> esp32 [port]
#   ./flash_and_capture.sh <model> pico  [port]
#   FQBN_ESP32/FQBN_PICO env vars override the board FQBNs.
set -euo pipefail

MODEL="${1:?model name, e.g. espcn_micro}"
BOARD="${2:?esp32|pico}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKETCH="$SCRIPT_DIR/../firmware/$MODEL"

case "$BOARD" in
  esp32)
    FQBN="${FQBN_ESP32:-esp32:esp32:esp32}"
    PORT_PATTERN='usbserial|SLAB|wchusb|cp210|ttyUSB'
    ;;
  pico)
    FQBN="${FQBN_PICO:-rp2040:rp2040:rpipico}"
    PORT_PATTERN='ttyACM|usbmodem'
    ;;
  *) echo "board must be esp32 or pico" >&2; exit 1 ;;
esac

PORT="${3:-$(arduino-cli board list | grep -iE "$PORT_PATTERN" | awk '{print $1}' | head -1)}"
[ -n "$PORT" ] || { echo "no port found for $BOARD" >&2; exit 1; }

echo "== $MODEL on $BOARD ($FQBN @ $PORT)"
arduino-cli compile --fqbn "$FQBN" "$SKETCH"
arduino-cli upload -p "$PORT" --fqbn "$FQBN" "$SKETCH"
sleep 2
python "$SCRIPT_DIR/capture_serial.py" "$PORT" "$MODEL" "$BOARD"
