# Contractility Analysis Tool

A desktop application for visualising and quantifying the contractile behaviour
of cardiac or smooth-muscle tissue samples measured in multi-well plates.
Inter-pole distance is tracked optically over time; the software derives beat
frequency and relative contraction from the raw pixel signal.

---

## Requirements
 
Requires **Python >= 3.10**. Create a virtual environment and install all
dependencies with the following commands:
 
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
 
# Install dependencies
pip install ttkbootstrap numpy matplotlib scipy
```

---

## Usage

1. Launch `gui.py`.
2. Click **CSV** in the header to load a measurement file.
3. Adjust the three control parameters in the left panel.
4. Click any well button on the right of a subplot to open a full-resolution
   detail view for that well.

### CSV format

```
t_0,  well_A1, well_A2, ..., well_H6
t_1,  ...
...
t_n,  ...
```

Column 0 is the time vector in seconds; every subsequent column is the
inter-pole distance signal of one well in pixels.  No header row.

---

## Controls

| Control | Unit | Effect |
|---|---|---|
| **Rows** | — | Number of subplots shown simultaneously (1–48) |
| **Max Frq** | Hz | Maximum expected beat frequency; sets minimum RR distance for peak detection |
| **Smooth** | samples | Moving-average kernel width for signal pre-smoothing |

Scroll-wheel **outside** the plot area scrolls through wells.
Scroll-wheel **inside** a subplot zooms the time axis around the cursor.

---

## Signal & Measurement Context

Each well contains a tissue sample suspended between two magnetic poles.
A camera records the distance between the poles over time.  When the tissue
contracts, the poles are pulled together and the pixel distance decreases.
Because the y-axis is displayed **inverted**, contractions appear as upward
peaks — consistent with the force-curve convention used in organ-bath systems.

```
large px value  →  poles far apart   →  tissue relaxed    (Diastole)
small px value  →  poles close       →  tissue contracted  (Systole)
```

---

## Analysis Pipeline

The full pipeline runs per well on every parameter change:

```
raw signal
    │
    ▼
smooth()          moving-average noise reduction
    │
    ▼
running_max()     diastolic baseline estimation  (runMax)
    │
    ▼
find_peaks()      systolic peak detection        (neigMin)
    │
    ▼
compute_metrics() peakHeight, contraction%, freq
```

---

## Mathematical Variables — Legend

### `t [s]` — Time vector
Uniformly sampled time axis from the CSV column 0.
Sampling interval: `dt = t[1] − t[0]`.

---

### `signal [px]` — Raw inter-pole distance
The unprocessed optical distance between the two poles in pixels.
Sign convention: decreasing value = contraction.

---

### `smoothed [px]` — Moving-average signal
Uniform-kernel convolution of width `w` (forced odd):

```
smoothed[i] = (1/w) * Σ signal[i−half : i+half+1]
```

Edges (`i < half` or `i ≥ n−half`) are filled with the original signal to
preserve array length.  Reduces high-frequency camera noise before all
subsequent steps.

---

### `runMax [px]` — Diastolic baseline

```
runMax[i] = max( smoothed[i − W/2 : i + W/2] )
```

A slow running maximum with window `W >> 1/f_beat`.  Because the window spans
several beat periods, only the relaxed (diastolic) envelope is captured — the
individual contractions fall below it and are invisible to the filter.
Displayed as the **gold line** in every subplot.

> Physiological meaning: the diastolic resting length of the tissue at time `i`.

---

### `neigMin [px]` — Systolic peak value

The signal value at each detected contraction peak — the local minimum within
a neighbourhood of half-width `min_spacing = floor(1 / maxFreq / dt)` samples.
Detected via a two-stage greedy algorithm (see `find_peaks`).
Displayed as **red dots**.

> Physiological meaning: the maximum shortening reached during each contraction.

---

### `peakHeight [px]` — Absolute contraction amplitude

```
peakHeight[i] = runMax[peak_i] − neigMin[i]    ≥ 0
```

The absolute pixel shortening of each contraction, measured from the diastolic
baseline at the moment of the peak.  Always non-negative because
`runMax ≥ smoothed` by definition.

> Physiological meaning: how far the tissue shortens in absolute pixel units
> during contraction `i`.

---

### `contraction [%]` — Relative contraction (Force[%])

```
contraction[i] = peakHeight[i] / runMax[peak_i] × 100
```

Normalises `peakHeight` to the diastolic baseline at that moment.  This makes
the value independent of the absolute inter-pole distance and therefore
comparable across wells with different resting lengths or camera offsets.

```
mean_contraction = mean( contraction[0..k] )
```

Displayed left of each subplot as **Force[%]**.

> Physiological meaning: the fractional shortening of the tissue relative to
> its diastolic length — analogous to the Frank-Starling fractional shortening
> metric used in classical cardiac physiology.

---

### `freq [Hz]` — Beat frequency

Derived from the mean RR interval rather than a simple peak count, which would
be distorted by spurious detections at the recording boundaries:

```
RR[i]   = t[ peak_idx[i+1] ] − t[ peak_idx[i] ]    [s]
meanRR  = mean( RR[0..k−2] )                         [s]
freq    = 1 / meanRR                                 [Hz]
```

Using the mean RR interval is equivalent to computing
`(k−1) / (t_last − t_first)`, which is the number of complete beat cycles
divided by the total time spanned — robust against double-peak artefacts that
would inflate `mean(1/RR)`.

Displayed left of each subplot in **Hz**.

---

## Plot Legend

| Visual | Meaning |
|---|---|
| Faint blue line | Raw inter-pole distance signal |
| Solid blue line | Smoothed signal |
| Gold line | Diastolic baseline (runMax) |
| Red dots | Detected systolic peaks (neigMin) |
| Left label top | Beat frequency [Hz] |
| Left label bottom | Mean relative contraction Force[%] |

## FOC48 - Pole Distance Analyzer

FOC48 analyzes contraction assays (e.g. cardiomyocyte measurements) on myrPlate well plates. For each of 48 wells (8 rows A-H x 6 columns 1-6), the distance between an electrode/pole pair is measured on every frame. As the tissue contracts and relaxes, the two poles of a pair move closer together and farther apart in a periodic pattern - this distance-over-time signal is the actual measurement output. The resulting time series is exported as CSV and serves as input for a separate, downstream peak-analysis tool (contraction frequency, amplitude, etc.).

Two input sources are supported:
- **avi**: a recorded video file
- **basler**: live acquisition from a Basler USB3 camera (Pylon SDK)
- **buffer**: a simulated rolling buffer fed from pre-extracted BMP frames (used to test the streaming pipeline without a camera)

---

## Installation (End Users)

1. Go to the "Releases" page of this repository: `https://github.com/<organization>/<repo>/releases`
2. Download the latest `.zip` file attached to the newest release.
3. Extract the `.zip` file to any location on your computer (e.g. `C:\FOC48`).

**Important:** how the archive is extracted matters. The `.zip` is built so that
`FOC48.exe`, `foc48.bat`, and all dependencies sit directly at the top level of
the archive. Some extraction tools create an extra nested folder around the
contents - make sure `FOC48.exe` and `foc48.bat` end up **directly inside**
your chosen folder (e.g. `C:\FOC48\FOC48.exe`), not one level deeper
(e.g. `C:\FOC48\FOC48\FOC48.exe`). If in doubt, extract first, then check
that `FOC48.exe` is visible right away without opening a further subfolder.

The extracted folder can be placed anywhere and moved later if needed - just
make sure `FOC48.exe` and `foc48.bat` always stay together in the same folder.

---

## How It Works

### 1. Calibration (once, on the first frame)

Before any distance measurements are taken, the first frame of a stream is used to calibrate the well grid:

1. **Global pole detection**: The frame is thresholded (Otsu's method) to separate bright poles from the dark background, then contours are extracted and filtered by area (`area_min`/`area_max`) to reject noise. Each valid contour's centroid is computed via image moments, yielding 96 individual pole positions (48 wells x 2 poles).

2. **Pairing**: The 96 detected poles are greedily paired by nearest-neighbor distance - the two closest unmatched points are paired first, then the next closest pair among the remaining points, and so on, until all poles are assigned to exactly one pair.

3. **Grid assignment**: The 48 pairs are sorted by their midpoint position (first by row/y-coordinate, then by column/x-coordinate within each row) and labeled A1-H6 in row-major order, matching the physical plate layout.

4. **Per-well ROI and threshold calibration**: For each pole individually, a small rectangular region of interest (ROI) is defined around its calibrated position, sized as a fraction (`local_margin_px`, default 0.45) of that pair's own measured distance. This factor is mathematically capped below 0.5, guaranteeing that `2 * margin < pair_distance` - i.e. the two ROI windows of a pair can never overlap, regardless of how close or far apart a given pair's poles are. Within each ROI, a local Otsu threshold is computed once and stored - this fixed, per-pole threshold (mirroring the original device software's `pixThreshL`/`pixThreshR` concept) is then reused for every subsequent frame, rather than recomputing a threshold each time.

### 2. Per-frame processing (repeated for every frame)

For each incoming frame, and for every well independently:

1. Both poles are located **only within their pre-calibrated ROI** (not the full image), using the pre-calibrated fixed threshold rather than a per-frame Otsu recalculation. This keeps processing fast and prevents a pole from one well being confused with a neighboring well.
2. If multiple blobs appear within a ROI (e.g. noise, or a neighboring pole intruding into the margin), the blob closest to the pole's calibrated template position is selected.
3. The pole's position is refined to sub-pixel accuracy via an intensity-weighted centroid (each pixel within the detected blob contributes to the centroid proportional to its grayscale intensity, rather than treating the blob as a uniform binary mask).
4. The Euclidean distance between the left and right pole is computed and recorded for that well and frame.

Frames are processed in parallel across multiple CPU cores (`ProcessPoolExecutor`), with the worker pool pre-warmed at startup to avoid measuring artificially low throughput during the first few frames while worker processes are still starting up.

---

## Requirements

### Installing the Basler Pylon Runtime

For `basler` mode, the Basler Pylon Runtime must be installed once on the target machine:

1. Open the download page: https://www.baslerweb.com/de-de/downloads/software/
2. Select the latest available version
3. Download the **Runtime Installer** (not the full "Software Suite" - that is not required)
4. Run the installer and follow the setup wizard
5. **Restart the PC** to ensure the environment variables set by the installer are reliably loaded

This step is only required **once per target machine**.

---

## Usage

### As a `.exe` (recommended for lab use)

After installing (see "Installation (End Users)" above) or building (see "Building the executable (for developers)" below), the folder contains the finished `FOC48.exe` alongside `foc48.bat`.

**Run:**

```
foc48.bat
```

Starts a default 60-second Basler live measurement at 60 fps.

**Analyze an AVI file:**

```
foc48.bat avi "C:\\path\\to\\video.avi"
```

**Creating a shortcut for convenient access:**

So that `foc48.bat` doesn't need to be located in the installation folder every time:

1. Right-click `foc48.bat` in the installation folder
2. Select "Create shortcut"
3. Move the resulting shortcut to the desktop (or any other location)

The shortcut points back to the original file - the batch file itself stays in place, so its relative path resolution to the `.exe` continues to work correctly.

### Directly via Python (development/debugging)

```bash
python main.py "video.avi" --mode avi
python main.py --mode basler --duration 60 --fps 60
python main.py data/test_bmps --mode buffer
```

All available parameters:

| Parameter | Description | Default |
|---|---|---|
| `input` | AVI file path (avi mode) or BMP directory (buffer mode); omit entirely in basler mode | none |
| `--mode` | `avi`, `basler`, `buffer` | `camera` |
| `--config` | Path to `config.yaml` | `config.yaml` next to `foc48.py` |
| `--output-dir` | Output directory for CSV, XML, first frame | `analyze_measurement` |
| `--fps` | Override for time column and acquisition rate | read from video/camera |
| `--workers` | Number of parallel worker processes | CPU count minus 1 |
| `--serial` | Basler camera serial number | first camera found |
| `--exposure-us` | Exposure time in microseconds (Basler) | camera default |
| `--duration` | Recording duration in seconds (Basler) | `60.0` |
| `--start-frame` / `--end-frame` | Frame range for AVI mode | entire video |

### Preparing `buffer` mode

The `buffer` mode does not read frames on its own - it consumes pre-extracted BMP frames from a directory. Before using `buffer` mode, frames must first be extracted from an AVI file with `extract_frames.py`:

```bash
python extract_frames.py "video.avi" --output-dir data/test_bmps
```

Optional arguments `--start-frame` and `--end-frame` limit the extracted range. Only after this step has produced BMP files in the target directory can `--mode buffer` be used to stream them through the pipeline.

### Identifying a specific Basler camera

Basler cameras are identified by their **serial number**, not by the physical USB port they are connected to - this way, measurements remain reproducible even if the camera is later connected to a different port.

If only one Basler camera is connected, no `--serial` argument is needed - it is selected automatically. If multiple cameras are connected simultaneously, list their serial numbers first:

```bash
python -c "from pypylon import pylon; [print(d.GetModelName(), d.GetSerialNumber()) for d in pylon.TlFactory.GetInstance().EnumerateDevices()]"
```

This prints the model name and serial number of every connected Basler camera, e.g.:

```
acA3088-16um 22945203
```

**Using it directly via Python:**

```bash
python main.py --mode basler --serial 22945203
```

**Using it in `foc48.bat`:**

Open `foc48.bat` in a text editor and add `--serial <number>` to the command line that starts `FOC48.exe`:

```batch
"%FOC48_EXE%" --mode basler --duration 60 --fps 60 --serial 22945203 --output-dir "%SCRIPT_DIR%analyze_measurement"
```

Replace `22945203` with the actual serial number identified in the step above.

---

## Configuration (`config.yaml`)

```yaml
analysis:
  n_rows: 8                   # Well rows (A-H)
  n_cols: 6                   # Well columns (1-6)
  area_min: 10                # Min. contour area for pole detection (px^2)
  area_max: 500                # Max. contour area for pole detection (px^2)
  invert_threshold: false      # true = dark poles on a bright background
  max_pair_distance: null      # Max. allowed distance for pairing (px), null = unlimited
  local_margin_px: null        # ROI margin factor, null = adaptive (0.45 x pair distance)
  fallback_pix_thresh: 20.0    # Fallback threshold if local Otsu calibration fails
  min_blob_area: 3             # Min. blob area during per-frame detection
  batch_size: 1                # Frames per processing task
  max_pending_batches: null    # Max. simultaneously buffered batches, null = automatic
  fps: 60.0                    # Fallback frame rate, if not otherwise specified
```

**Key parameters explained:**

- `area_min`/`area_max`: Must match the actual pole size in the image. If detection fails (too few/too many poles found), adjust these first.
- `local_margin_px`: Determines the size of each pole's local search window. The adaptive default (`null`) computes this automatically from the measured pair distance, mathematically guaranteeing the two ROI windows of a pair never overlap.
- `batch_size`: For small, fast processing steps like this one, `1` typically yields the best throughput, since workload is distributed most evenly across worker processes.

---

## Output Format

Each run produces three files (naming scheme: `<source>_output.csv`, `<source>_calibration.xml`, `<source>0.bmp`):

### CSV (`<name>_output.csv`)

| time | A1 | A2 | ... | H6 |
|---|---|---|---|---|
| 0.000000 | 108.42 | 95.31 | ... | 112.05 |
| 0.016667 | 107.88 | 94.77 | ... | 111.62 |

- **`time`**: timestamp in seconds, computed from the frame rate
- **`A1`...`H6`**: measured pole-pair distance in pixels for that well
- Missing detections (e.g. a pole temporarily not found) are recorded as `NaN`

### Calibration XML (`<name>_calibration.xml`)

Contains the calibration results for each well:

```xml
<calibration>
  <meta>
    <source>basler_live</source>
  </meta>
  <A1>
    <pixThreshL>22.500</pixThreshL>
    <pixThreshR>21.800</pixThreshR>
    <roiLeft>(120, 85, 195, 160)</roiLeft>
    <roiRight>(210, 85, 285, 160)</roiRight>
    <templateLeft>(157.3, 122.1)</templateLeft>
    <templateRight>(247.8, 121.9)</templateRight>
  </A1>
  ...
</calibration>
```

- **`pixThreshL`/`pixThreshR`**: calibrated, fixed thresholds for the left/right pole
- **`roiLeft`/`roiRight`**: bounding box of the local search window, in image coordinates
- **`templateLeft`/`templateRight`**: reference pole position at the time of calibration

### First Frame (`<name>0.bmp`)

Reference image of the very first acquired/read frame, always saved regardless of mode - used for traceability of the calibration.

---

## Building the executable (for developers)

> This section is only relevant for developers who need to (re-)build `FOC48.exe`. End users running the finished application on a lab PC do not need Python, PyInstaller, or the source code - only the contents of the release `.zip` (see "Installation (End Users)" above).

`build.bat` compiles the Python source into a standalone executable using PyInstaller.

**Requirements on the build machine:**
- Python 3.10 or newer
- All project dependencies (`opencv-python`, `numpy`, `pandas`, `PyYAML`, `pypylon`)
- PyInstaller (installed automatically by `build.bat` if missing)

**Run from the project root** (the same folder as `main.py`):

```bash
build.bat
```

This will:
1. Verify the installed Python version
2. Install PyInstaller if not already present
3. Remove any previous build artifacts (`build/`, `dist/`, `FOC48.spec`)
4. Bundle `main.py` and all required modules (`calibration.py`, `processing.py`, `stream_source.py`), including `pypylon`'s native binaries, into `dist/FOC48/`

**After building**, copy `foc48.bat` into `dist/FOC48/` alongside `FOC48.exe`, then zip the **contents** of `dist/FOC48/` (not the folder itself, to avoid an extra nested folder) and upload the `.zip` as a release asset:

```bash
Compress-Archive -Path "dist\FOC48\*" -DestinationPath "FOC48_vX.Y.Z.zip"
```

Publish it via the repository's "Releases" page (`.../releases/new`) rather than as a regular committed file, since GitHub's regular file/commit upload has a much lower size limit than release assets.
"""
```