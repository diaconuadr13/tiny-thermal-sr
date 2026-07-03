#!/usr/bin/env python3
"""Reset an ESP32/Pico via DTR/RTS and capture one fresh boot of a tiny-thermal-sr sketch.

The sketches print once in setup() and leave loop() empty, so the monitor
must trigger a reset after attaching. Reads until "CAS:end" (or timeout) and
writes the log to measure/logs/<model>_<board>.log while echoing it.

Usage:
    python measure/capture_serial.py /dev/ttyUSB0 espcn_micro esp32
    python measure/capture_serial.py /dev/ttyACM0 espcn_micro pico [secs]
"""
import sys
import time
from pathlib import Path

import serial

LOG_DIR = Path(__file__).resolve().parent / "logs"

port = sys.argv[1]
model = sys.argv[2]
board = sys.argv[3]
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0

ser = serial.Serial()
ser.port = port
ser.baudrate = 115200
ser.timeout = 0.2
ser.dtr = False  # GPIO0 high -> normal boot (run app, not bootloader)
ser.rts = False
ser.open()

# Auto-reset "run" pulse (ESP32: EN via RTS; Pico ACM ports just reopen).
ser.dtr = False
ser.rts = True
time.sleep(0.12)
ser.reset_input_buffer()
ser.rts = False

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR / f"{model}_{board}.log"
lines = []
t0 = time.time()
done = False
while time.time() - t0 < secs:
    raw = ser.readline()
    if not raw:
        if done:
            break
        continue
    s = raw.decode("utf-8", "replace")
    sys.stdout.write(s)
    sys.stdout.flush()
    lines.append(s)
    if "CAS:end" in s or "CAS:status=NOFIT" in s or "CAS:status=ERR" in s:
        done = True
ser.close()

log_path.write_text("".join(lines))
sys.stderr.write(f"[capture] {time.time() - t0:.1f}s, done={done} -> {log_path}\n")
