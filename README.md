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