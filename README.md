# 🎛️ HOT-Step Yue2 LoRA Training Parameter Calculator

![Python](https://img.shields.io/badge/python-3.12%20tested-blue?logo=python&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-Tkinter-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![Status](https://img.shields.io/badge/status-experimental-yellow)

A desktop tool that **analyses a folder of training audio** and recommends **HOT-Step LoRA training settings** for Yue2 loras. After a trial run you give it **feedback**, and you can **calibrate** it on the settings that actually worked, so its recommendations for new datasets get closer to what makes good LoRAs.

> [!NOTE]
> The recommendations come from heuristic formulas, not from a trained model. Treat them as a **starting point** and use the calibration tab to tune them to your own results. It has already been calibrated twice & since I do not have access to all variations of all datasets, I cannot make one set of calibration files to fit all. This provides a good starting point based on the specifics of the provided dataset.  

---

## 📑 Contents

- [🟦 Features](#-features)
- [🟩 Installation](#-installation)
- [🟧 Quick start](#-quick-start)
- [🟪 How it works](#-how-it-works)
- [🟥 Factors that drive the settings](#-factors-that-drive-the-settings)
- [🟫 Calibration in detail](#-calibration-in-detail)
- [⬜ Files and notes](#-files-and-notes)

---

## 🟦 Features

- 🔎 **Dataset analysis** of `.wav`, `.flac` & `.mp3` files, with chunked files grouped back into their parent tracks
- 🏷️ **Caption check** that reads **only** `<trackname>.yue2.txt` files (one line of comma-separated tags)
- 🧮 **Recommended settings** for rank, alpha, steps, learning rate, planner scale, timing loss weight, caption dropout, decay steps, decay shape and accumulation
- 🔁 **Feedback loop** using **Instruments** and **Vocals** verdicts (Weak / Good / Overcooked) and the best peak step
- ✍️ **Adjust From** boxes: enter the settings you actually ran and the feedback corrects **those** values, not the calculated ones
- 📐 **Calibration tab**: save a run that worked as a **datum point**; the calculator is then re-anchored on it for new datasets
- ♻️ **Reset button** that clears your entries, the feedback and the calibration to get back to the plain calculation
- ✅ **Valid ranks only**: **8, 16, 32, 64, 128, 256** (alpha equals rank)

---

## 🟩 Installation

### Python packages

```bash
pip install numpy librosa soundfile
```

| Needed | Used for | Where it comes from |
|---|---|---|
| **`numpy`** | the DSP maths | pip |
| **`librosa`** | spectral flux, spectral centroid, zero-crossing rate and RMS analysis | pip (pulls in scipy, numba, scikit-learn and others automatically) |
| **`soundfile`** | reading each file's duration and format | pip (the Windows wheel bundles libsndfile) |
| **`tkinter`** | the window | bundled with the python.org Windows installer. In a virtual environment it works if the base install had "tcl/tk" ticked. |
| **`lora_calibration.py`** | the Calibration tab | included in this repo, in the **same folder** as `settingsv83cal.py`. It needs only the standard library. |

Everything else (`os`, `sys`, `math`, `json`, `re`, `csv`, `shutil`, `time`) ships with Python.

> [!TIP]
> There is **no GPU, PyTorch or model download** required.

- **Python version:** tested on **3.12**. Anything from 3.8 up is expected to work but has not been checked.
- **Missing packages:** if one of the three pip packages is missing, the calculator shows the exact `pip install` line when you press **Analyze**.
- **MP3 support:** needs a recent `soundfile` (**0.12 or newer**). If MP3s fail to read, run `pip install -U soundfile`.
- **Without `lora_calibration.py`:** the calculator still runs and the Calibration tab just says it is disabled.

---

## 🟧 Quick start

```bash
python Settings_v85cal.py
```

1. Click **Browse Folder...** and choose your dataset folder.
2. Choose your **Optimizer** and **Accumulation Steps**, then press **Analyze Dataset & Calculate Dynamic Settings**.
3. Train with the recommended settings.
4. Enter the **Best Peak Step** and rate the **Instruments** and **Vocals**, and the settings update.
5. To correct your **own** tweaked settings instead, press **Copy calculated settings into these boxes** (or type them in), edit as needed, then give feedback.
6. Once a run produces a good LoRA, record it on the **Calibration** tab (see [Calibration in detail](#-calibration-in-detail)).

### 🏷️ Caption files

Each caption lives next to its audio file and is a **single line of comma-separated tags**:

```
song.wav
song.yue2.txt   ->   synthpop, melodic house, female lead vocal, bright detuned lead synth, ..., 124 BPM
```

Matching is case-insensitive. No other sidecar type (`.txt`, `.json` and so on) is read.

---

## 🟪 How it works

1. 🔬 **Scan the dataset.** Audio files are found and chunked files (`_chunk1`, `_part2`, `_slice3`, `_clip4`) are grouped into parent tracks. **10-second clips** are sampled from each track and measured with librosa.
2. 🏷️ **Read the captions.** Each `<trackname>.yue2.txt` is split on commas to count tags. A track with no caption counts as **0 tags**.
3. 📊 **Score complexity.** The audio measurements are combined into an **audio complexity** score. A **caption richness factor** (**0.80 to 1.30**, based on average tags per track) multiplies it to give the **combined complexity** score.
4. 🧮 **Calculate the base settings:**
   - **Steps** grow with the log of track count, complexity and duration.
   - **Learning rate** rises with track count and complexity, then is divided by the square root of the accumulation steps.
   - **Rank** follows track count and complexity, is capped by dataset size, and snaps to **8 / 16 / 32 / 64 / 128 / 256**. Alpha equals rank.
   - **Planner scale** and **timing loss weight** follow the audio analysis.
   - **Caption dropout** is **0.50** for chunked or small datasets, otherwise **0.35**.
   - **Schedule** is a stable phase followed by a **40% decay**, with the **decay shape** (Cosine or Linear) chosen from the spectral variance.
5. 🔁 **Apply feedback after a trial run.** You rate instruments and vocals, and you can enter the best checkpoint step.
   - **Instruments Overcooked:** rank down, dropout to 0.50, longer decay.
   - **Instruments Weak:** rank and learning rate up.
   - **Vocals Overcooked:** learning rate down, dropout to 0.50.
   - **Vocals Weak:** rank up.
   - **Best peak step:** re-fits the total steps so the peak sits at the end of the stable phase.
6. ✍️ **Adjust from your own settings.** Any value you enter in the **Adjust From** boxes replaces the calculated starting value for that setting, and the corrections are applied on top. Blank boxes use the calculated value. When rank changes, it moves **one valid step** (for example 64 to 128) per correction.
7. 📐 **Calibrate.** Save a run that worked as a datum point (see below). On that dataset the calculator then reproduces your settings exactly. On other datasets it keeps its own curve, shifted to start from your datum.

> [!IMPORTANT]
> While your own settings are in the **Adjust From** boxes, the active calibration is **not** applied. Your values are the base.

---

## 🟥 Factors that drive the settings

| Group | Factors |
|---|---|
| 🟦 **Dataset** | number of parent tracks, number of chunks, total duration |
| 🟩 **Audio features** | spectral flux, spectral centroid variation, zero-crossing rate, RMS energy variation |
| 🟨 **Captions** | tags per track in `.yue2.txt` files, averaged over **all** tracks (**4 tags or fewer = 0.80, 28 tags or more = 1.30**) |
| 🟧 **Your choices** | optimizer (**Prodigy** and **Muon** use the engine's default learning rate), accumulation steps, decay-shape override |
| 🟪 **Feedback** | instrument verdict, vocal verdict, best peak step |
| 🟫 **Your own settings** | the **Adjust From** boxes |
| 🟥 **Calibration** | the active datum point |

### What the caption richness factor changes

From a sparse caption (4 tags or fewer) to a full one (28 tags), measured on simulated datasets:

| Setting | Effect |
|---|---|
| **Learning rate** | about **+28%** |
| **Rank** | raw capacity up about **62%**, but only changes when it crosses a valid rank step |
| **Steps and decay steps** | about **+8% to +15%** |
| **Planner scale, timing loss weight, caption dropout, decay shape** | **unchanged** |

The curve is a heuristic and **needs real-world testing**. To change it, edit `CAPTION_TAGS_MIN` and `CAPTION_TAGS_FULL` near the top of the calculation code in `Ssettings_v87cal_EXP.py`. The total number of captions that can influence settings was increased. 

---

## 🟫 Calibration in detail

The calculator's plain figures will not always match what works for your model and music. Calibration records the difference and uses it for new datasets.

1. **Analyze** the dataset you trained, so calibration knows what the calculator recommended for it.
2. Open the **Calibration** tab and enter **all** the settings of the run that **worked**: rank, alpha, timing loss weight, total steps, learning rate, planner scale, caption dropout, decay steps, decay shape, accumulation and optimizer.
3. Enter the analyser results: **best checkpoint step**, singer score, vibe score and the Instruments and Vocals verdicts. **Import evaluator CSV** can fill the best step and scores from `checkpoint_eval_results.csv`.
4. Give it a **Run name** and press **Save as calibration point**. The dataset name from the front tab is saved in a column.
5. Select the point in the table and press **Use selected point as calibration**. It becomes the active **datum**, marked with ●, and its name shows on both tabs and in the output.

**What the datum does:** for each setting it stores the ratio between what worked and what the plain calculator said (dropout is shifted by the difference instead). New datasets get the calculator's own result multiplied by those ratios. On the datum's own dataset that gives back exactly what you entered, including **total steps** and **decay steps**. The learning rate is stored before accumulation scaling, so it stays comparable if you change accumulation.

**Buttons:** Fill from current settings, Import evaluator CSV, Save as calibration point, Use selected point as calibration, Stop using calibration, Delete selected point, **Remove ALL calibration**.

Points are stored in **`calibration.cfg`**, created in the folder you start the program from. A corrupt file is backed up as `calibration.cfg.corrupt` rather than overwritten.

---

## ⬜ Files and notes

| File | Purpose |
|---|---|
| `Settings_v85cal.py` | the calculator and GUI |
| `Settings_v87cal_EXP.py` | the experimental version of the calculated adjustments for captions - the total number of captions provided to the model for setting adjustments was increased |
| `lora_calibration.py` | calibration storage and maths (standard library only) |
| `calibration.cfg` | created at runtime in the folder you launch from |

- The calculation logic (for captions) and GUI were tested on simulated dataset values in Settings_v87cal_EXP version . Real-world testing is still needed.
- The **Clear boxes + reset to standard calculation** button returns to the plain calculation and switches calibration off. Tick **Apply the active calibration** to turn it back on.
