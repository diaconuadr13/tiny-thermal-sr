# tiny-thermal-sr

Fully integer-quantized (INT8) thermal image super-resolution on
microcontrollers — ESP32 and Raspberry Pi Pico. Codebase for the CAS 2026
paper (imt.ro/cas); grew out of the `sisr-embedded-systems` thesis repo.

**Task:** grayscale thermal SR, LR 32×24 → HR 64×48 (×2), fully-integer INT8
(int8 weights, activations AND I/O — hybrid models are rejected by esp-nn
TFLite Micro on ESP32).

**Key design decision:** models are written and trained in **Keras** with a
restricted op vocabulary (`Conv2D`, `ReLU`, `depth_to_space`, `Add`), so the
trained graph converts to full-INT8 TFLite by construction — no
PyTorch→Keras porting, no parity checks, no hybrid ops possible.

**Ground truth:** FLIR-IISR HR frames, grayscale, whole-frame resized to
64×48 (HR) / 32×24 (LR). Real PSNR/SSIM. The native MLX90640 dataset
(32×24) has **no** ground truth at 64×48 and is used **qualitatively only**
(demo figures + firmware test frames) — never PSNR against bicubic pseudo-HR.

## Models

| name | origin | INT8-friendly modifications |
|---|---|---|
| `espcn` | Shi 2016 | Tanh → ReLU |
| `espcn_light` | thesis | — (already ReLU) |
| `espcn_micro` | thesis | — (already ReLU) |
| `fsrcnn_ds` | Dong 2016 | PReLU → ReLU; 9×9 deconv → 3×3 conv + depth_to_space |
| `edsr_tiny` | Lim 2017 (reduced) | res_scale 0.1 folded into conv weights at export |

## Setup

```bash
pip install -r requirements.txt
# on the training server, the thesis venv already has everything:
#   source /scratch/users/adiaconu/sisr-embedded-systems/.disertatie/bin/activate
```

Raw source datasets (FLIR-IISR, MLX90640/Zenodo) live in the thesis repo;
point `--thesis-repo` / `CAS_THESIS_REPO` at its checkout for step 1 only.
Everything downstream is self-contained here.

## Pipeline (run in order, from repo root)

```bash
# 1. Data: FLIR-IISR -> data/flir_thermal_x2, MLX90640 demo frames
python data_prep/prepare_data.py --thesis-repo /path/to/sisr-embedded-systems

# 2. Train all 5 (CPU, ~minutes each)
for m in espcn_micro espcn_light espcn fsrcnn_ds edsr_tiny; do
    python train_tf.py --config configs/$m.yaml
done

# 3. Full-INT8 export + verification + quality eval + C headers
python export/export_int8.py

# 4. Generate firmware (one sketch per model, works on BOTH boards via #ifdef)
python measure/make_firmware.py

# 5. Flash each firmware/<model>/ sketch (machine with boards + arduino-cli):
./measure/flash_and_capture.sh espcn_micro esp32
./measure/flash_and_capture.sh espcn_micro pico
# Pico note: RP2040 USB-CDC drops output unless the host holds DTR asserted,
# and there is no reset line - start capture_pico.py BEFORE `arduino-cli
# upload` (it retry-opens through re-enumeration):
#   python measure/capture_pico.py /dev/cu.usbmodemXXXX <model> pico 300 &
#   arduino-cli upload -p /dev/cu.usbmodemXXXX --fqbn rp2040:rp2040:rpipico firmware/<model>
python measure/parse_serial_log.py            # -> results/board_metrics.csv

# 6. Aggregate + figures
python results/collect_results.py             # -> results/main_table.csv
python figures/make_qualitative.py
python figures/make_scatter.py
```

## Boards

- **ESP32-DevKitC V4 (no PSRAM)** — Arduino + Espressif TFLite Micro
  (esp-nn). Arena allocated at runtime from internal heap, ladder-down on
  failure. EDSR-Tiny is expected **not to fit** — that is a reported result.
- **Raspberry Pi Pico (RP2040)** — Arduino (earlephilhower core) +
  pico-tflmicro. Static arena.

Firmware prints `CAS:key=value` at 115200 baud: model/arena/heap sizes,
first + 10 sampled `Invoke()` times, **bit-exactness check** of the int8
output against the PC interpreter reference embedded in `test_image.h`, and
an SR output dump.

## Results (val split, real GT; board columns pending measurements)

| model | params | flash KB | PSNR f32 | PSNR int8 | drop dB |
|---|---|---|---|---|---|
| espcn_micro | 1.6K | 5.2 | 30.92 | 30.75 | 0.17 |
| espcn_light | 6.0K | 10.1 | 31.45 | 30.81 | 0.64 |
| fsrcnn_ds | 10.1K | 20.8 | 31.61 | 31.13 | 0.49 |
| espcn | 21.3K | 26.2 | 31.73 | 31.45 | 0.28 |
| edsr_tiny | 158.7K | 185.2 | 32.83 | 32.58 | 0.25 |

Bicubic baseline: 30.44 dB. Full table: `results/main_table.csv`.

## Honesty notes (for the paper)

- Energy figures are **estimated** (datasheet chip-level active current ×
  measured latency), labeled as such — no power meter was used.
- MLX90640 SR results are qualitative; quantitative claims use the
  FLIR-derived val split only.
- On-device outputs are NOT universally bit-exact vs the PC interpreter, but
  the deviation is bounded and identical on both boards (ESPCN family: <=3.9%
  of pixels by +-1 LSB; FSRCNN-ds: 31% by up to +-4; EDSR-Tiny: 49% by up to
  +-3) - deterministic kernel rounding, so PC quality transfers within
  sub-quantization-step bounds. Per-model numbers: results/board_metrics.csv.
- espcn_light's 0.64 dB PTQ drop is intrinsic (calibration-size independent);
  QAT is currently blocked by tfmot × Keras 3 incompatibility.
