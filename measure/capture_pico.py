#!/usr/bin/env python3
"""Capture a CAS sketch run on the Raspberry Pi Pico (USB-CDC serial).

Differences vs capture_serial.py (which targets the ESP32's UART bridge):
  * Pico USB-CDC drops all output unless the host holds the port open with
    DTR asserted, so we open with dtr=True — never False.
  * There is no reset line: the sketch runs once right after the UF2 upload
    reboots the board. So START THIS SCRIPT FIRST (it retry-opens until the
    port re-enumerates), THEN run `arduino-cli upload` — the sketch's 2 s
    startup delay gives the retry loop time to attach.

Usage (note the ordering):
    python measure/capture_pico.py /dev/cu.usbmodem1101 <model> pico [secs] &
    arduino-cli upload -p /dev/cu.usbmodem1101 --fqbn rp2040:rp2040:rpipico firmware/<model>
    wait
"""
import sys
import time
from pathlib import Path

import serial

LOG_DIR = Path(__file__).resolve().parent / "logs"

port = sys.argv[1]
model = sys.argv[2]
board = sys.argv[3]
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 600.0

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR / f"{model}_{board}.log"
lines = []
t0 = time.time()
done = False
ser = None

def try_open():
    s = serial.Serial()
    s.port = port
    s.baudrate = 115200
    s.timeout = 0.2
    s.dtr = True   # Pico CDC: host must assert DTR or prints are dropped
    s.rts = True
    s.open()
    return s

while time.time() - t0 < secs and not done:
    if ser is None:
        try:
            ser = try_open()
            sys.stderr.write(f"[capture-pico] port open at t+{time.time()-t0:.1f}s\n")
        except (serial.SerialException, OSError):
            time.sleep(0.15)   # port gone (bootloader phase) — retry
            continue
    try:
        raw = ser.readline()
    except (serial.SerialException, OSError):
        # port vanished mid-read (upload re-enumeration) — reopen
        try:
            ser.close()
        except Exception:
            pass
        ser = None
        continue
    if not raw:
        continue
    s = raw.decode("utf-8", "replace")
    sys.stdout.write(s)
    sys.stdout.flush()
    lines.append(s)
    if "CAS:end" in s or "CAS:status=NOFIT" in s or "CAS:status=ERR" in s:
        done = True

if ser is not None:
    ser.close()
log_path.write_text("".join(lines))
sys.stderr.write(f"[capture-pico] {time.time()-t0:.1f}s, done={done} -> {log_path}\n")
sys.exit(0 if done else 1)
