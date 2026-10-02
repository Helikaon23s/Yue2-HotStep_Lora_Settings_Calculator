import os
import sys
import math
import json
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# --- Calibration logic (lora_calibration.py must sit in the same folder) ---
CAL_IMPORT_ERROR = ""
try:
    import lora_calibration as cal
except Exception as _cal_err:
    cal = None
    CAL_IMPORT_ERROR = str(_cal_err)

# --- PRE-FLIGHT DEPENDENCY CHECKER ---
MISSING_DEPS = []

try:
    import numpy as np
except ImportError:
    MISSING_DEPS.append("numpy")

try:
    import librosa
except ImportError:
    MISSING_DEPS.append("librosa")

try:
    import soundfile as sf
except ImportError:
    MISSING_DEPS.append("soundfile")


def check_dependencies_and_warn():
    """Warns the user via GUI if required packages are missing before executing."""
    if MISSING_DEPS:
        dep_str = ", ".join(MISSING_DEPS)
        install_cmd = f"pip install {' '.join(MISSING_DEPS)}"
        msg = (
            f"Missing required Python libraries:\n\n  • {dep_str}\n\n"
            f"Please run the following command in your terminal/venv to fix:\n\n"
            f"  {install_cmd}"
        )
        messagebox.showerror("Missing Dependencies", msg)
        return False
    return True


def parse_mm3_sidecar(filepath):
    """Parses MiniMax-Music 3 (.mm3.txt) metadata files."""
    LYRIC_KEYS = {'lyrics', 'lyric', 'vocal_text', 'words', 'text'}
    field_details = {}
    total_words = 0

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            in_lyrics_block = False

            for line in lines:
                clean_line = line.strip()
                if not clean_line:
                    continue

                if ':' in clean_line:
                    prefix, val = clean_line.split(':', 1)
                    prefix_key = prefix.strip().lower()
                    val_clean = val.strip()

                    if prefix_key in LYRIC_KEYS:
                        in_lyrics_block = True
                        continue
                    else:
                        in_lyrics_block = False
                        field_details[prefix_key] = val_clean
                        total_words += len(val_clean.split())
                        continue

                if not in_lyrics_block:
                    total_words += len(clean_line.split())
    except Exception:
        pass

    return total_words, len(field_details)


def extract_caption_from_file(filepath):
    """Extracts caption text from standard text or JSON sidecars."""
    lower_path = filepath.lower()
    if lower_path.endswith('.bak') or lower_path.endswith('.lyrics.txt') or lower_path.endswith('.caption.json') or lower_path.endswith('.mm3.txt'):
        return ""

    ext = os.path.splitext(filepath)[1].lower()
    CAPTION_KEYS = {'caption', 'prompts', 'prompt', 'description', 'tags', 'genre', 'summary', 'style'}
    LYRIC_KEYS = {'lyrics', 'lyric', 'vocal_text', 'words', 'text'}

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            if ext in ('.json', '.jsonl'):
                line = f.readline().strip() if ext == '.jsonl' else f.read()
                data = json.loads(line) if line else {}

                if isinstance(data, dict):
                    for k, v in data.items():
                        if k.lower() in CAPTION_KEYS and isinstance(v, str):
                            return v.strip()
                    extracted_parts = []
                    for k, v in data.items():
                        if k.lower() not in LYRIC_KEYS:
                            if isinstance(v, str):
                                extracted_parts.append(v)
                            elif isinstance(v, list):
                                extracted_parts.append(" ".join(str(x) for x in v))
                    return " ".join(extracted_parts).strip()
                elif isinstance(data, list):
                    return " ".join(str(x) for x in data)
            else:
                lines = f.readlines()
                caption_lines = []
                in_lyrics_block = False

                for line in lines:
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                    if ':' in clean_line:
                        prefix, val = clean_line.split(':', 1)
                        prefix_lower = prefix.strip().lower()
                        if prefix_lower in CAPTION_KEYS:
                            caption_lines.append(val.strip())
                            in_lyrics_block = False
                            continue
                        elif prefix_lower in LYRIC_KEYS:
                            in_lyrics_block = True
                            continue
                    if not in_lyrics_block:
                        caption_lines.append(clean_line)

                return " ".join(caption_lines).strip()
    except Exception:
        return ""

    return ""


def extract_base_track_info(filename):
    """Extracts the root track identifier, grouping audio chunks to parent tracks."""
    stem = os.path.splitext(filename)[0]
    pattern = re.compile(r'^(.*?)[_\-\s]*(?:chunk|part|slice|clip)[_\-\s]*(\d+)', re.IGNORECASE)
    match = pattern.match(stem)
    if match:
        return match.group(1).strip('_- '), int(match.group(2))
    return stem, 0


YUE2_SUFFIX = ".yue2.txt"


def analyze_caption_sidecars(folder_path, parent_track_bases, is_mm3_mode=False):
    """Caption density from <track>.yue2.txt files ONLY (one line of plain caption, e.g. mysong.yue2.txt next to mysong.wav).
    Matching is case-insensitive; no other sidecar type (.txt/.json/.mm3.txt ...) is looked at."""
    try:
        listing = {name.lower(): name for name in os.listdir(folder_path)}
    except OSError:
        listing = {}

    caption_word_counts = []
    tag_counts = []
    tags_all = []
    files_found = 0
    for base_name in parent_track_bases:
        words = 0
        n_tags = 0
        actual = listing.get((base_name + YUE2_SUFFIX).lower())
        if actual:
            files_found += 1
            try:      # .yue2.txt = one long line of comma-separated tags: count words and tags
                with open(os.path.join(folder_path, actual), 'r', encoding='utf-8', errors='ignore') as fh:
                    text = fh.read()
                words = len(text.split())
                if words:
                    n_tags = len([t for t in re.split(r'[,\n]+', text) if t.strip()])
                    tag_counts.append(n_tags)
            except OSError:
                words = 0
        caption_word_counts.append(words)
        tags_all.append(n_tags)

    avg_words = float(np.mean(caption_word_counts)) if caption_word_counts else 0.0
    sidecar_count = sum(1 for w in caption_word_counts if w > 0)

    return {
        "avg_word_count": avg_words,
        "captioned_tracks": sidecar_count,
        "yue2_files": files_found,
        "avg_tag_count": float(np.mean(tag_counts)) if tag_counts else 0.0,
        "avg_tags_all": float(np.mean(tags_all)) if tags_all else 0.0,
        "caption_ratio": sidecar_count / max(1, len(parent_track_bases))
    }


def analyze_dataset_dsp(folder_path, is_mm3_mode=False, clip_duration_sec=10.0):
    """Executes librosa DSP analysis across sampled slices of dataset tracks."""
    valid_exts = ('.wav', '.flac', '.mp3', '.ogg')
    raw_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]

    if not raw_files:
        return None

    parent_groups = {}
    total_audio_duration = 0.0

    for filename in raw_files:
        filepath = os.path.join(folder_path, filename)
        try:
            info = sf.info(filepath)
            dur = info.duration
            total_audio_duration += dur
        except Exception:
            continue

        base_name, chunk_idx = extract_base_track_info(filename)
        if base_name not in parent_groups:
            parent_groups[base_name] = []

        parent_groups[base_name].append({'path': filepath, 'duration': dur, 'chunk_idx': chunk_idx})

    c_std_list, flux_list, zcr_list, rms_std_list = [], [], [], []

    for base_name, chunks in parent_groups.items():
        chunks.sort(key=lambda x: x['chunk_idx'])
        parent_duration = sum(c['duration'] for c in chunks)
        if parent_duration <= 0:
            continue

        target_positions = [parent_duration * 0.10, parent_duration * 0.50, parent_duration * 0.80]
        track_centroids, track_fluxes, track_zcrs, track_rms_stds = [], [], [], []

        for pos in target_positions:
            accum_time = 0.0
            target_chunk = None
            local_offset = 0.0

            for c in chunks:
                if accum_time <= pos < accum_time + c['duration']:
                    target_chunk = c
                    local_offset = pos - accum_time
                    break
                accum_time += c['duration']

            if not target_chunk:
                target_chunk = chunks[-1]
                local_offset = max(0.0, target_chunk['duration'] - clip_duration_sec)

            if local_offset + clip_duration_sec > target_chunk['duration']:
                local_offset = max(0.0, target_chunk['duration'] - clip_duration_sec)

            try:
                y, sr = librosa.load(target_chunk['path'], sr=22050, mono=True, offset=local_offset, duration=clip_duration_sec)
                if len(y) == 0:
                    continue

                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                track_centroids.append(float(np.std(centroid)))

                stft = np.abs(librosa.stft(y))
                flux = np.mean(np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0)))
                track_fluxes.append(float(flux))

                zcr = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])
                track_zcrs.append(float(zcr))

                rms = librosa.feature.rms(y=y)[0]
                track_rms_stds.append(float(np.std(rms)))
            except Exception:
                continue

        if track_centroids:
            c_std_list.append(np.mean(track_centroids))
            flux_list.append(np.mean(track_fluxes))
            zcr_list.append(np.mean(track_zcrs))
            rms_std_list.append(np.mean(track_rms_stds))

    if not c_std_list:
        return None

    parent_track_bases = list(parent_groups.keys())
    caption_info = analyze_caption_sidecars(folder_path, parent_track_bases)

    return {
        "num_chunks": len(raw_files),
        "num_parent_tracks": len(parent_track_bases),
        "total_duration_secs": total_audio_duration,
        "c_std": float(np.mean(c_std_list)),
        "spectral_flux": float(np.mean(flux_list)),
        "zcr": float(np.mean(zcr_list)),
        "rms_std": float(np.mean(rms_std_list)),
        "avg_caption_words": caption_info["avg_word_count"],
        "captioned_tracks": caption_info["captioned_tracks"],
        "yue2_files": caption_info["yue2_files"],
        "avg_caption_tags": caption_info["avg_tag_count"],
        "avg_tags_all": caption_info["avg_tags_all"],
        "caption_ratio": caption_info["caption_ratio"]
    }


# Caption richness is measured in comma-separated tags (<track>.yue2.txt). A caption this long counts as "full".
CAPTION_TAGS_MIN = 4       # at or below this many tags the caption adds no complexity (factor 0.80)
CAPTION_TAGS_FULL = 28     # your longest caption: factor reaches its maximum (1.30)
CALC_VERSION = 2           # bump when the plain formula changes; calibration points record it

VALID_RANKS = (8, 16, 32, 64, 128, 256)


def _snap_rank(x):
    """Nearest valid LoRA rank (8/16/32/64/128/256), measured in log2."""
    return min(VALID_RANKS, key=lambda r: abs(math.log2(r) - math.log2(max(float(x), 1.0))))


def _step_rank(base, direction):
    """Move `direction` (+1 / -1 / 0) valid ranks away from the valid rank nearest `base`."""
    i = VALID_RANKS.index(_snap_rank(base)) + direction
    return VALID_RANKS[max(0, min(len(VALID_RANKS) - 1, i))]


def calculate_hotstep_pertinent_settings(
    num_tracks: int,
    num_chunks: int,
    total_duration_secs: float,
    c_std: float,
    spectral_flux: float,
    zcr: float,
    rms_std: float,
    avg_caption_words: float,
    override_accum: int = 1,
    optimizer_choice: str = "AdamW (graph, experimental)",
    prior_best_step: int = 0,
    inst_feedback: str = "Good",
    vocal_feedback: str = "Good",
    decay_shape_mode: str = "Auto",
    calib_mult: dict = None,
    base_override: dict = None,
    avg_caption_tags: float = None
):
    """Calculates all recommended parameters using DSP features + qualitative feedback.

    base_override  {"rank","lr","steps","accum","dropout","alpha","timing","planner","decay_steps","decay_shape"}
                   = the settings YOU are running.  They replace the calculated starting values for EVERY setting
                   they cover, and the feedback corrections are applied on top of them.
    calib_mult     anchor from the active calibration datum (see lora_calibration.anchor_from_point);
                   ignored while base_override is used.
    """
    use_override = bool(base_override)
    base_override = base_override or {}
    calib = {} if use_override else (calib_mult or {})

    norm_cstd = max(0.0, min(c_std / 900.0, 1.0))
    norm_flux = max(0.0, min(spectral_flux / 3.5, 1.0))
    norm_zcr = max(0.0, min(zcr / 0.18, 1.0))
    norm_rms = max(0.0, min(rms_std / 0.15, 1.0))

    audio_complexity = 0.60 + ((norm_cstd * 0.45) + (norm_flux * 0.35) + (norm_zcr * 0.20)) * 1.0
    if avg_caption_tags is not None:
        tag_t = max(0.0, min((avg_caption_tags - CAPTION_TAGS_MIN) / float(CAPTION_TAGS_FULL - CAPTION_TAGS_MIN), 1.0))
        caption_density_factor = 0.80 + 0.50 * tag_t
    else:
        caption_density_factor = max(0.80, min(0.75 + (avg_caption_words / 60.0), 1.30))
    combined_complexity = audio_complexity * caption_density_factor

    accumulation = max(1, override_accum)
    duration_hours = max(0.05, total_duration_secs / 3600.0)

    # 1. Base step calculation
    base_steps = 150 + (130 * math.log(max(2, num_tracks) * combined_complexity * (1.0 + 0.20 * math.log1p(duration_hours))))
    pure_total_steps = int(round((base_steps * 1.15) / 10.0) * 10)
    pure_peak_step = int(round(pure_total_steps * (1.0 - 0.40)))
    baseline_total_steps = pure_total_steps
    if calib.get("steps"):                      # calibration datum: reproduce the run length that worked (no rounding)
        baseline_total_steps = max(10, int(round(pure_total_steps * calib["steps"])))
    elif calib.get("peak"):
        baseline_total_steps = max(10, int(round(pure_total_steps * calib["peak"] / 10.0) * 10))

    rank_multiplier = 1.0
    lr_multiplier = 1.0
    decay_ratio = calib.get("decay_ratio", 0.40)
    dropout_override = None
    reasons = []

    # 2. Process feedback modifiers
    if inst_feedback == "Overcooked":
        rank_multiplier *= 0.75
        dropout_override = 0.50
        decay_ratio += 0.05
        reasons.append("Reduced Rank & increased Dropout to fix overcooked/memorized instruments")
    elif inst_feedback == "Weak":
        rank_multiplier *= 1.25
        lr_multiplier *= 1.15
        reasons.append("Increased Rank & Learning Rate to boost weak instrument fidelity")

    if vocal_feedback == "Overcooked":
        dropout_override = 0.50
        lr_multiplier *= 0.85
        reasons.append("Lowered Learning Rate & boosted Dropout to prevent vocal over-fitting")
    elif vocal_feedback == "Weak":
        rank_multiplier *= 1.20
        reasons.append("Increased Rank capacity to capture missing vocal nuances")

    # 3. Comprehensive DSP-Driven Decay Shape Analysis
    dsp_variance_score = (norm_flux * 0.45) + (norm_cstd * 0.35) + (norm_rms * 0.20)

    if decay_shape_mode == "Auto":
        if base_override.get("decay_shape") in ("Cosine", "Linear"):
            final_decay_shape = base_override["decay_shape"]
            reasons.append(f"Decay shape kept at your entered {final_decay_shape}")
        elif calib.get("decay_shape") in ("Cosine", "Linear"):
            final_decay_shape = calib["decay_shape"]
            reasons.append(f"Auto (Calibration): {final_decay_shape} - the shape used in your calibration run")
        elif prior_best_step > 0:
            final_decay_shape = "Cosine"
            reasons.append("Auto (DSP + Calibration): Cosine selected for smooth harmonic convergence on recalibrated step schedule.")
        elif dsp_variance_score > 0.60:
            final_decay_shape = "Linear"
            reasons.append(f"Auto (DSP Analysis): Linear selected due to high spectral variance ({dsp_variance_score:.2f}). Gives a steady cooling rate across mixed transients.")
        else:
            final_decay_shape = "Cosine"
            reasons.append(f"Auto (DSP Analysis): Cosine selected due to consistent spectral profile ({dsp_variance_score:.2f}). Preserves delicate timbral details.")
    else:
        final_decay_shape = decay_shape_mode
        reasons.append(f"Decay shape manually overridden to {final_decay_shape}")

    # 4. Step Schedule Calculation
    if prior_best_step > 0:
        stable_ratio = 1.0 - decay_ratio
        calculated_total = prior_best_step / stable_ratio
        total_steps = int(round(calculated_total / 10.0) * 10)
        decay_steps = int(round((total_steps * decay_ratio) / 10.0) * 10)
        reasons.append(f"Calibrated WSD schedule around prior peak step {prior_best_step}")
    elif base_override.get("steps"):
        total_steps = int(base_override["steps"])
        if base_override.get("decay_steps") and abs(decay_ratio - 0.40) < 1e-9:
            decay_steps = int(base_override["decay_steps"])
        else:
            decay_steps = int(round((total_steps * decay_ratio) / 10.0) * 10)
        reasons.append(f"Steps start from your entered {total_steps} (enter a Best Peak Step to re-fit the schedule)")
    else:
        total_steps = baseline_total_steps
        if "decay_ratio" in calib:
            decay_steps = int(round(total_steps * decay_ratio))
        else:
            decay_steps = int(round((total_steps * decay_ratio) / 10.0) * 10)

    # 5. Learning Rate Calculation
    sigmoid_capacity = 1.0 / (1.0 + math.exp(-(num_tracks - 35) / 12.0))
    pure_unscaled_lr = (2.8e-5 + (6.2e-5 * sigmoid_capacity)) * math.sqrt(combined_complexity) * 1.35
    if base_override.get("lr"):
        old_accum = max(1, base_override.get("accum") or accumulation)
        unscaled_start = base_override["lr"] * math.sqrt(old_accum)       # undo the entered run's accumulation scaling
        target_unscaled_lr = unscaled_start * lr_multiplier
    else:
        target_unscaled_lr = pure_unscaled_lr * calib.get("lr", 1.0) * lr_multiplier
    scaled_lr = target_unscaled_lr / math.sqrt(accumulation)

    if "Prodigy" in optimizer_choice or "Muon" in optimizer_choice:
        lr_display = "engine default"
    else:
        lr_display = f"{scaled_lr:.6f}"

    # 6. LoRA Rank Calculation (valid ranks only: 8/16/32/64/128/256)
    log_volume_scale = math.log2(max(2.0, num_tracks / 3.0))
    pure_rank_capacity = (5.5 + (10.0 * log_volume_scale)) * combined_complexity
    max_rank_allowed = max(32.0, min(256.0, 16.0 * math.log2(max(2, num_tracks))))
    pure_rank_raw = max(8.0, min(pure_rank_capacity, max_rank_allowed))
    pure_rank = _snap_rank(pure_rank_raw)

    if base_override.get("rank"):
        typed = base_override["rank"]
        direction = 1 if rank_multiplier > 1.0001 else (-1 if rank_multiplier < 0.9999 else 0)
        rank = _step_rank(typed, direction)
        if typed not in VALID_RANKS:
            reasons.append(f"Rank {typed} is not a valid rank (8/16/32/64/128/256) - treated as {_snap_rank(typed)}")
        if direction:
            reasons.append(f"Rank moved one valid step: {_snap_rank(typed)} -> {rank} (valid ranks: 8/16/32/64/128/256)")
    else:
        raw_rank = max(8.0, min(pure_rank_capacity * rank_multiplier, max_rank_allowed)) * calib.get("rank", 1.0)
        rank = _snap_rank(raw_rank)

    alpha_ratio = calib.get("alpha_ratio", 1.0)
    if base_override.get("alpha") and base_override.get("rank"):
        alpha_ratio = base_override["alpha"] / base_override["rank"]
    alpha = max(1, int(round(rank * alpha_ratio)))

    planner_pure = round(max(0.20, min(0.30 + (audio_complexity - 0.6) * 0.25, 0.50)), 2)
    timing_pure = round(max(0.05, min(0.08 + (norm_flux * 0.04), 0.15)), 2)
    planner_scale, timing_loss_weight = planner_pure, timing_pure
    if base_override.get("planner") is not None:
        planner_scale = base_override["planner"]
    elif calib.get("planner"):
        planner_scale = round(planner_pure * calib["planner"], 3)
    if base_override.get("timing") is not None:
        timing_loss_weight = base_override["timing"]
    elif calib.get("timing"):
        timing_loss_weight = round(timing_pure * calib["timing"], 3)

    is_chunked = num_chunks > num_tracks
    default_dropout = 0.50 if (is_chunked or num_tracks < 15) else 0.35
    base_dropout = base_override.get("dropout")
    if base_dropout is None:
        base_dropout = default_dropout
        if "dropout_delta" in calib:
            base_dropout = round(min(0.9, max(0.0, default_dropout + calib["dropout_delta"])), 2)
    if dropout_override is not None:
        if use_override and base_dropout >= 0.50:
            caption_dropout = min(0.70, round(base_dropout + 0.10, 2))    # already high: keep pushing, capped
        else:
            caption_dropout = max(base_dropout, dropout_override)
    else:
        caption_dropout = base_dropout

    if use_override:
        reasons.insert(0, "Starting from YOUR entered settings (Adjust From boxes), not the calculated baseline; blank boxes use the calculated value")

    return {
        "rank": rank,
        "alpha": alpha,
        "timing_loss_weight": timing_loss_weight,
        "steps": total_steps,
        "optimizer": optimizer_choice,
        "learning_rate": lr_display,
        "planner_scale": planner_scale,
        "caption_dropout": caption_dropout,
        "decay_steps": decay_steps,
        "decay_shape": final_decay_shape,
        "accumulation": accumulation,
        "audio_complexity": audio_complexity,
        "combined_complexity": combined_complexity,
        "caption_density_factor": caption_density_factor,
        "dsp_variance_score": dsp_variance_score,
        "reasons": reasons,
        # --- values used by calibration ---
        "lr_unscaled": target_unscaled_lr,
        "pure_unscaled_lr": pure_unscaled_lr,
        "pure_rank": pure_rank,
        "pure_rank_raw": pure_rank_raw,
        "pure_peak_step": pure_peak_step,
        "pure_total_steps": pure_total_steps,
    }


class HOTStepCalculatorApp:
    CAL_FIELDS = [("rank", "Rank"), ("alpha", "Alpha"), ("timing_loss_weight", "Timing loss wt"),
                  ("steps", "Steps (total)"), ("learning_rate", "Learning rate"), ("planner_scale", "Planner LR scale"),
                  ("caption_dropout", "Caption dropout"), ("decay_steps", "Decay steps"),
                  ("decay_shape", "Decay shape"), ("accumulation", "Accumulation")]

    def __init__(self, root):
        self.root = root
        self.root.title("HOT-Step Training Parameter Calculator")
        self.root.geometry("1040x1040")

        self.folder_path = tk.StringVar()
        self.optimizer_choice = tk.StringVar(value="AdamW (graph, experimental)")
        self.accum_override = tk.StringVar(value="1")
        self.prior_best_step = tk.StringVar(value="0")

        self.inst_feedback_var = tk.StringVar(value="Good")
        self.vocal_feedback_var = tk.StringVar(value="Good")
        self.decay_shape_mode_var = tk.StringVar(value="Auto")

        # dataset context + calibration
        self.genre_var = tk.StringVar()
        self.singers_var = tk.StringVar(value="1")
        self.use_cal_var = tk.BooleanVar(value=True)
        self.cal_active_text = tk.StringVar()

        # "Adjust From": the settings you are running (all of the recommended settings)
        self.af_fields = list(self.CAL_FIELDS)
        self.af = {k: tk.StringVar() for k, _ in self.af_fields}
        self.af["accumulation"].set("1")
        self.af_status = tk.StringVar()

        self.cal_path = cal.cfg_path() if cal else None      # calibration.cfg lives where the program was started
        self.cal_data = cal.load(self.cal_path) if cal else {"points": [], "active": None}

        self.last_dsp_results = None
        self.last_config = None
        self.last_pure = None
        self._create_widgets()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _num(text, cast=float):
        try:
            return cast(str(text).strip())
        except (ValueError, TypeError):
            return None

    def _backend(self):
        return "MiniMax-Music 3" if (self.last_dsp_results or {}).get("is_mm3") else "HOT-Step"

    def _dataset_name(self):
        return os.path.basename(self.folder_path.get().strip().rstrip("/\\"))

    # ------------------------------------------------------------------ UI
    def _create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        tab_calc = ttk.Frame(self.notebook)
        tab_cal = ttk.Frame(self.notebook)
        self.notebook.add(tab_calc, text="  Calculator  ")
        self.notebook.add(tab_cal, text="  Calibration  ")

        main_frame = ttk.Frame(tab_calc, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Folder Selection
        path_frame = ttk.LabelFrame(main_frame, text=" Dataset Directory ", padding="10")
        path_frame.pack(fill=tk.X, pady=(0, 10))

        entry_path = ttk.Entry(path_frame, textvariable=self.folder_path, font=("Consolas", 10))
        entry_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        btn_browse = ttk.Button(path_frame, text="Browse Folder...", command=self.browse_folder)
        btn_browse.pack(side=tk.RIGHT)

        # 2. Hardware & Schedule Settings
        opts_frame = ttk.LabelFrame(main_frame, text=" Hardware & Schedule Settings ", padding="10")
        opts_frame.pack(fill=tk.X, pady=(0, 10))

        lbl_opt = ttk.Label(opts_frame, text="Optimizer:")
        lbl_opt.grid(row=0, column=0, sticky=tk.W, padx=(5, 5), pady=5)

        combo_opt = ttk.Combobox(
            opts_frame, textvariable=self.optimizer_choice,
            values=["AdamW (graph, experimental)", "AdamW", "Prodigy", "Muon"],
            state="readonly", width=25
        )
        combo_opt.grid(row=0, column=1, sticky=tk.W, padx=(0, 20), pady=5)
        combo_opt.bind("<<ComboboxSelected>>", lambda e: self.on_options_changed())

        lbl_accum = ttk.Label(opts_frame, text="Accumulation Steps:")
        lbl_accum.grid(row=0, column=2, sticky=tk.W, padx=(5, 5), pady=5)

        entry_accum = ttk.Spinbox(
            opts_frame, from_=1, to=64, textvariable=self.accum_override, width=6
        )
        entry_accum.grid(row=0, column=3, sticky=tk.W, padx=(0, 10), pady=5)
        entry_accum.bind("<KeyRelease>", lambda e: self.on_options_changed())
        entry_accum.bind("<<Increment>>", lambda e: self.on_options_changed())
        entry_accum.bind("<<Decrement>>", lambda e: self.on_options_changed())

        # Decay Shape Selection Mode
        lbl_shape = ttk.Label(opts_frame, text="Decay Shape Selector:")
        lbl_shape.grid(row=1, column=0, sticky=tk.W, padx=(5, 5), pady=5)

        shape_subframe = ttk.Frame(opts_frame)
        shape_subframe.grid(row=1, column=1, columnspan=3, sticky=tk.W)

        rb_shape_auto = ttk.Radiobutton(shape_subframe, text="Auto (DSP Analyzed)", variable=self.decay_shape_mode_var, value="Auto", command=self.on_options_changed)
        rb_shape_cos = ttk.Radiobutton(shape_subframe, text="Cosine Override", variable=self.decay_shape_mode_var, value="Cosine", command=self.on_options_changed)
        rb_shape_lin = ttk.Radiobutton(shape_subframe, text="Linear Override", variable=self.decay_shape_mode_var, value="Linear", command=self.on_options_changed)

        rb_shape_auto.pack(side=tk.LEFT, padx=(0, 15))
        rb_shape_cos.pack(side=tk.LEFT, padx=(0, 15))
        rb_shape_lin.pack(side=tk.LEFT)

        # Genre / singers / calibration toggle
        ttk.Label(opts_frame, text="Genre / music type:").grid(row=2, column=0, sticky=tk.W, padx=(5, 5), pady=5)
        self.genre_combo = ttk.Combobox(opts_frame, textvariable=self.genre_var, width=25, values=self._known_genres())
        self.genre_combo.grid(row=2, column=1, sticky=tk.W, padx=(0, 20), pady=5)
        self.genre_combo.bind("<<ComboboxSelected>>", lambda e: self.on_options_changed())
        self.genre_combo.bind("<KeyRelease>", lambda e: self.on_options_changed())

        ttk.Label(opts_frame, text="Singers in dataset:").grid(row=2, column=2, sticky=tk.W, padx=(5, 5), pady=5)
        sp_singers = ttk.Spinbox(opts_frame, from_=1, to=50, textvariable=self.singers_var, width=6)
        sp_singers.grid(row=2, column=3, sticky=tk.W, padx=(0, 10), pady=5)
        sp_singers.bind("<KeyRelease>", lambda e: self.on_options_changed())
        sp_singers.bind("<<Increment>>", lambda e: self.on_options_changed())
        sp_singers.bind("<<Decrement>>", lambda e: self.on_options_changed())

        chk_cal = ttk.Checkbutton(opts_frame, text="Apply the active calibration (Calibration tab)",
                                  variable=self.use_cal_var, command=self.on_options_changed)
        chk_cal.grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=(5, 5), pady=(0, 2))
        ttk.Label(opts_frame, textvariable=self.cal_active_text, foreground="#1a6e1a").grid(
            row=4, column=0, columnspan=4, sticky=tk.W, padx=(5, 5), pady=(0, 5))

        # 2b. Adjust From: the settings you are actually running
        start_frame = ttk.LabelFrame(main_frame, text=" Adjust From - the settings you are running (the feedback below corrects THESE) ", padding="8")
        start_frame.pack(fill=tk.X, pady=(0, 10))

        keys = [k for k, _ in self.af_fields] + ["optimizer"]
        labels = dict(self.af_fields)
        labels["optimizer"] = "Optimizer"
        labels["accumulation"] = "Accum (that run)"
        for idx, key in enumerate(keys):
            r, c = divmod(idx, 4)
            ttk.Label(start_frame, text=labels[key] + ":").grid(row=r, column=c * 2, sticky=tk.W, padx=(5, 3), pady=3)
            if key == "optimizer":
                w = ttk.Combobox(start_frame, textvariable=self.optimizer_choice, width=14, state="readonly",
                                 values=["AdamW (graph, experimental)", "AdamW", "Prodigy", "Muon"])
                w.bind("<<ComboboxSelected>>", lambda e: self.on_options_changed())
            elif key == "decay_shape":
                w = ttk.Combobox(start_frame, textvariable=self.af[key], width=9, values=["", "Cosine", "Linear"])
                w.bind("<<ComboboxSelected>>", lambda e: self.on_options_changed())
                w.bind("<KeyRelease>", lambda e: self.on_options_changed())
            else:
                w = ttk.Entry(start_frame, textvariable=self.af[key], width=9)
                w.bind("<KeyRelease>", lambda e: self.on_options_changed())
            w.grid(row=r, column=c * 2 + 1, sticky=tk.W, padx=(0, 10), pady=3)

        af_btns = ttk.Frame(start_frame)
        af_btns.grid(row=3, column=0, columnspan=8, sticky=tk.W, pady=(6, 2))
        ttk.Button(af_btns, text="Copy calculated settings into these boxes",
                   command=self.copy_calculated_to_adjust_from).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(af_btns, text="Clear boxes + reset to standard calculation (no calibration)",
                   command=self.reset_to_standard).pack(side=tk.LEFT)
        ttk.Label(start_frame, textvariable=self.af_status, wraplength=950, justify=tk.LEFT).grid(
            row=4, column=0, columnspan=8, sticky=tk.W, padx=5)

        # 3. Prior Run Quality Feedback Frame
        feedback_frame = ttk.LabelFrame(main_frame, text=" Prior Run Calibration Feedback (Leave default for 1st Run) ", padding="10")
        feedback_frame.pack(fill=tk.X, pady=(0, 10))

        lbl_prior = ttk.Label(feedback_frame, text="Best Peak Step (0 = First Run):")
        lbl_prior.grid(row=0, column=0, sticky=tk.W, padx=(5, 5), pady=5)

        entry_prior = ttk.Spinbox(
            feedback_frame, from_=0, to=10000, increment=10, textvariable=self.prior_best_step, width=10
        )
        entry_prior.grid(row=0, column=1, sticky=tk.W, padx=(5, 20), pady=5)
        entry_prior.bind("<KeyRelease>", lambda e: self.on_options_changed())
        entry_prior.bind("<<Increment>>", lambda e: self.on_options_changed())
        entry_prior.bind("<<Decrement>>", lambda e: self.on_options_changed())

        # Instrument Radio Group
        lbl_inst = ttk.Label(feedback_frame, text="Instruments Sound:")
        lbl_inst.grid(row=1, column=0, sticky=tk.W, padx=(5, 5), pady=5)

        inst_subframe = ttk.Frame(feedback_frame)
        inst_subframe.grid(row=1, column=1, columnspan=3, sticky=tk.W)

        rb_inst_over = ttk.Radiobutton(inst_subframe, text="Exact Match / Overcooked", variable=self.inst_feedback_var, value="Overcooked", command=self.on_options_changed)
        rb_inst_good = ttk.Radiobutton(inst_subframe, text="Close / Good Timbre", variable=self.inst_feedback_var, value="Good", command=self.on_options_changed)
        rb_inst_weak = ttk.Radiobutton(inst_subframe, text="Lacks Character / Weak", variable=self.inst_feedback_var, value="Weak", command=self.on_options_changed)

        rb_inst_over.pack(side=tk.LEFT, padx=(0, 10))
        rb_inst_good.pack(side=tk.LEFT, padx=(0, 10))
        rb_inst_weak.pack(side=tk.LEFT)

        # Vocal Radio Group
        lbl_vocal = ttk.Label(feedback_frame, text="Vocals Sound:")
        lbl_vocal.grid(row=2, column=0, sticky=tk.W, padx=(5, 5), pady=5)

        vocal_subframe = ttk.Frame(feedback_frame)
        vocal_subframe.grid(row=2, column=1, columnspan=3, sticky=tk.W)

        rb_voc_over = ttk.Radiobutton(vocal_subframe, text="Exact Match / Overcooked", variable=self.vocal_feedback_var, value="Overcooked", command=self.on_options_changed)
        rb_voc_good = ttk.Radiobutton(vocal_subframe, text="Close / Good Timbre", variable=self.vocal_feedback_var, value="Good", command=self.on_options_changed)
        rb_voc_weak = ttk.Radiobutton(vocal_subframe, text="Lacks Character / Weak", variable=self.vocal_feedback_var, value="Weak", command=self.on_options_changed)
        rb_voc_na = ttk.Radiobutton(vocal_subframe, text="N/A (Instrumental)", variable=self.vocal_feedback_var, value="NA", command=self.on_options_changed)

        rb_voc_over.pack(side=tk.LEFT, padx=(0, 10))
        rb_voc_good.pack(side=tk.LEFT, padx=(0, 10))
        rb_voc_weak.pack(side=tk.LEFT, padx=(0, 10))
        rb_voc_na.pack(side=tk.LEFT)

        # Action Button
        btn_analyze = tk.Button(
            main_frame, text="Analyze Dataset & Calculate Dynamic Settings",
            bg="#2b8cbe", fg="white", font=("Helvetica", 11, "bold"),
            pady=8, command=self.run_analysis
        )
        btn_analyze.pack(fill=tk.X, pady=(0, 15))

        # Output Text Area
        output_frame = ttk.LabelFrame(main_frame, text=" Dataset Analysis & Recommended Settings ", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)

        out_scroll = ttk.Scrollbar(output_frame, orient=tk.VERTICAL)
        out_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_output = tk.Text(output_frame, font=("Consolas", 10), wrap=tk.WORD, bg="#1e1e1e", fg="#00ffcc",
                                  yscrollcommand=out_scroll.set)
        self.txt_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        out_scroll.config(command=self.txt_output.yview)

        self._build_calibration_tab(tab_cal)
        self.folder_path.trace_add("write", lambda *a: self.cal_refresh_view())
        self.cal_refresh_view()

    # ------------------------------------------------------------------ calibration tab
    def _known_genres(self):
        return sorted({p.get("genre", "") for p in self.cal_data.get("points", []) if p.get("genre")})

    def _active_point(self):
        if cal is None:
            return None
        return cal.get_point(self.cal_data, self.cal_data.get("active"))

    def _build_calibration_tab(self, parent):
        if cal is None:
            ttk.Label(parent, padding=20, wraplength=700,
                      text="lora_calibration.py was not found next to this script, so calibration is disabled.\n"
                           f"({CAL_IMPORT_ERROR})").pack(anchor=tk.W)
            return

        frm = ttk.Frame(parent, padding="12")
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, wraplength=980, justify=tk.LEFT, text=(
            "1) Analyze the dataset you trained on the Calculator tab (calibration needs to know what the calculator "
            "recommended for it).\n"
            "2) Enter ALL the settings of the run that WORKED plus the analyser results (best checkpoint step, scores), "
            "give it a Run name, and Save.\n"
            "3) Select a saved point and press 'Use selected point'. It becomes the datum: the calculator is re-anchored "
            "so that, for new datasets, it starts from the settings that worked here instead of its own figures.")
                  ).pack(anchor=tk.W, pady=(0, 6))

        self.cal_name = tk.StringVar()
        self.cal_dataset_text = tk.StringVar()
        self.cv = {k: tk.StringVar() for k, _ in self.CAL_FIELDS}
        self.cv["accumulation"].set("1")
        self.cv["optimizer"] = tk.StringVar(value=self.optimizer_choice.get())
        for k in ("best_step", "singer", "vibe"):
            self.cv[k] = tk.StringVar()
        self.cal_inst = tk.StringVar(value="Good")
        self.cal_vocal = tk.StringVar(value="Good")

        run = ttk.LabelFrame(frm, text=" The run that worked ", padding="8")
        run.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(run, text="Run name:").grid(row=0, column=0, sticky=tk.W, padx=(5, 3), pady=3)
        ttk.Entry(run, textvariable=self.cal_name, width=26).grid(row=0, column=1, columnspan=3, sticky=tk.W, padx=(0, 10))
        ttk.Label(run, textvariable=self.cal_dataset_text).grid(row=0, column=4, columnspan=4, sticky=tk.W, padx=5)

        keys = [k for k, _ in self.CAL_FIELDS] + ["optimizer"]
        labels = dict(self.CAL_FIELDS)
        labels["optimizer"] = "Optimizer"
        for idx, key in enumerate(keys):
            r, c = divmod(idx, 4)
            ttk.Label(run, text=labels[key] + ":").grid(row=r + 1, column=c * 2, sticky=tk.W, padx=(5, 3), pady=3)
            if key == "optimizer":
                w = ttk.Combobox(run, textvariable=self.cv[key], width=14, state="readonly",
                                 values=["AdamW (graph, experimental)", "AdamW", "Prodigy", "Muon"])
            elif key == "decay_shape":
                w = ttk.Combobox(run, textvariable=self.cv[key], width=9, values=["", "Cosine", "Linear"])
            else:
                w = ttk.Entry(run, textvariable=self.cv[key], width=9)
            w.grid(row=r + 1, column=c * 2 + 1, sticky=tk.W, padx=(0, 10), pady=3)

        met = ttk.LabelFrame(frm, text=" Analyser results for that run ", padding="8")
        met.pack(fill=tk.X, pady=(0, 6))
        for idx, (key, label) in enumerate([("best_step", "BEST checkpoint step"), ("singer", "Singer score"), ("vibe", "Vibe score")]):
            ttk.Label(met, text=label + ":").grid(row=0, column=idx * 2, sticky=tk.W, padx=(5, 3), pady=3)
            ttk.Entry(met, textvariable=self.cv[key], width=9).grid(row=0, column=idx * 2 + 1, sticky=tk.W, padx=(0, 14))
        ttk.Label(met, text="Instruments:").grid(row=1, column=0, sticky=tk.W, padx=(5, 3), pady=3)
        ttk.Combobox(met, textvariable=self.cal_inst, values=["Good", "Weak", "Overcooked", "NA"], width=11,
                     state="readonly").grid(row=1, column=1, sticky=tk.W)
        ttk.Label(met, text="Vocals:").grid(row=1, column=2, sticky=tk.W, padx=(5, 3), pady=3)
        ttk.Combobox(met, textvariable=self.cal_vocal, values=["Good", "Weak", "Overcooked", "NA"], width=11,
                     state="readonly").grid(row=1, column=3, sticky=tk.W)

        b1 = ttk.Frame(frm)
        b1.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(b1, text="Fill from current settings", command=self.cal_fill_from_current).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(b1, text="Import evaluator CSV (best step + scores)", command=self.cal_import_csv).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(b1, text="Save as calibration point", bg="#2b8cbe", fg="white", font=("Helvetica", 10, "bold"),
                  command=self.cal_save_point).pack(side=tk.LEFT)

        cols = ("active", "name", "dataset", "genre", "tracks", "rank", "alpha", "steps", "lr", "best", "singer", "vibe")
        widths = (45, 135, 135, 75, 50, 45, 45, 55, 80, 50, 55, 55)
        self.cal_tree = ttk.Treeview(frm, columns=cols, show="headings", height=7, selectmode="browse")
        for c, w in zip(cols, widths):
            self.cal_tree.heading(c, text=c)
            self.cal_tree.column(c, width=w, anchor=tk.W)
        self.cal_tree.pack(fill=tk.X, pady=(4, 4))

        b2 = ttk.Frame(frm)
        b2.pack(fill=tk.X, pady=(0, 6))
        tk.Button(b2, text="Use selected point as calibration", bg="#1a7f37", fg="white", font=("Helvetica", 10, "bold"),
                  command=self.cal_use_selected).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(b2, text="Stop using calibration", command=self.cal_stop_using).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(b2, text="Delete selected point", command=self.cal_delete_selected).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(b2, text="Remove ALL calibration", command=self.cal_remove_all).pack(side=tk.LEFT)

        self.cal_status = tk.StringVar()
        ttk.Label(frm, textvariable=self.cal_status, wraplength=980, justify=tk.LEFT).pack(anchor=tk.W)

    def cal_refresh_view(self):
        if cal is None or not hasattr(self, "cal_tree"):
            return
        active = self.cal_data.get("active")
        self.cal_tree.delete(*self.cal_tree.get_children())
        for p in self.cal_data["points"]:
            ex, w = p.get("extras", {}), p.get("worked", {})
            self.cal_tree.insert("", tk.END, iid=p["id"], values=(
                "\u25cf" if p["id"] == active else "", p["name"], p.get("dataset", ""), p.get("genre", ""),
                p["features"].get("n_tracks", ""), w.get("rank", ""), w.get("alpha", ""), ex.get("steps", ""),
                ex.get("lr_entered", ""), ex.get("best_step", ""), ex.get("singer", ""), ex.get("vibe", "")))
        self.genre_combo["values"] = self._known_genres()
        self.cal_dataset_text.set(f"Dataset (from front tab): {self._dataset_name() or '(none selected)'}")
        ap = self._active_point()
        if ap:
            label = f"'{ap['name']}' (dataset: {ap.get('dataset') or '?'})"
            if self.use_cal_var.get():
                self.cal_active_text.set(f"Calibration datum in use: {label}")
                state = f"ACTIVE CALIBRATION: {label}"
            else:
                self.cal_active_text.set(f"Calibration datum {label} is SWITCHED OFF (tick the box above to use it)")
                state = f"Active point {label} - currently switched off on the Calculator tab."
        else:
            self.cal_active_text.set("No calibration datum active - plain calculator.")
            state = "No calibration active - the plain calculator is used."
        self.cal_status.set(f"File: {self.cal_path}\n{len(self.cal_data['points'])} saved point(s).  {state}")

    def _dataset_features(self):
        dsp, pure = self.last_dsp_results, self.last_pure
        if not dsp or not pure:
            return None
        feats = {"n_tracks": dsp["num_parent_tracks"], "total_minutes": dsp["total_duration_secs"] / 60.0,
                 "combined_complexity": pure["combined_complexity"]}
        singers = self._num(self.singers_var.get(), int)
        if singers:
            feats["n_singers"] = singers
        return feats

    def _current_run_settings(self):
        """The settings in the Adjust From boxes; blank ones are filled from the calculated settings."""
        c = self.last_config
        out = {}
        for k, _ in self.af_fields:
            v = self.af[k].get().strip()
            if not v and c and k in c:
                v = str(c[k])
                if k == "learning_rate" and self._num(v) is None:
                    v = ""
            out[k] = v
        out["optimizer"] = self.optimizer_choice.get()
        return out if (c or any(out.values())) else None

    def cal_fill_from_current(self):
        run = self._current_run_settings()
        if not run:
            messagebox.showinfo("Calibration", "Analyze a dataset on the Calculator tab first.")
            return
        for k, _ in self.CAL_FIELDS:
            self.cv[k].set(run.get(k, ""))
        self.cv["optimizer"].set(run.get("optimizer", self.optimizer_choice.get()))

    def cal_import_csv(self):
        f = filedialog.askopenfilename(title="Evaluator results CSV (checkpoint_eval_results.csv)",
                                       filetypes=[("CSV", "*.csv")], initialdir=os.getcwd())
        if not f:
            return
        try:
            r = cal.import_evaluator_csv(f)
        except Exception as e:
            messagebox.showerror("Import failed", str(e))
            return
        if not self.cv["steps"].get().strip():
            self.cv["steps"].set(str(r["total_steps"]))
        self.cv["best_step"].set(str(r["best_step"]))
        self.cv["singer"].set(str(r["singer"]))
        if r["vibe"] is not None:
            self.cv["vibe"].set(str(r["vibe"]))

    def cal_save_point(self):
        if not self.last_dsp_results or not self.last_pure:
            messagebox.showwarning("Calibration", "Analyze the dataset on the Calculator tab first - "
                                   "calibration needs to know what the calculator recommended for it.")
            return

        def f(k):
            return self._num(self.cv[k].get(), float)

        rank, alpha, lr, steps, best = f("rank"), f("alpha"), f("learning_rate"), f("steps"), f("best_step")
        accum = int(f("accumulation") or 1)
        if rank and int(rank) not in VALID_RANKS:
            messagebox.showwarning("Calibration", "Rank must be one of 8, 16, 32, 64, 128, 256.")
            return
        if not (rank or lr or best or steps):
            messagebox.showwarning("Calibration", "Enter the settings that worked (at least rank, learning rate, steps or best step).")
            return

        pure = self.last_pure
        baseline = {"lr": pure["pure_unscaled_lr"], "rank": pure["pure_rank_raw"], "steps": pure["pure_total_steps"],
                    "peak": pure["pure_peak_step"],
                    "timing": pure["timing_loss_weight"], "planner": pure["planner_scale"],
                    "dropout": pure["caption_dropout"]}
        worked = {}
        if rank:
            worked["rank"] = int(rank)
            if alpha:
                worked["alpha"] = int(alpha) if float(alpha).is_integer() else alpha
        if lr:
            worked["lr"] = lr * math.sqrt(accum)          # undo accumulation scaling so it compares like-for-like
        decay_steps_typed = f("decay_steps")
        if steps:                                       # the run length that worked is what gets reproduced
            worked["steps"] = int(round(steps))
            if decay_steps_typed:
                worked["decay_steps"] = int(round(decay_steps_typed))
        elif best:                                      # no total steps typed: fall back to the best checkpoint
            worked["peak"] = int(round(best))
        if f("timing_loss_weight") is not None:
            worked["timing"] = f("timing_loss_weight")
        if f("planner_scale") is not None:
            worked["planner"] = f("planner_scale")
        if f("caption_dropout") is not None:
            worked["dropout"] = f("caption_dropout")
        shape = self.cv["decay_shape"].get().strip().capitalize()
        if shape in ("Cosine", "Linear"):
            worked["decay_shape"] = shape

        extras = {k: self.cv[k].get().strip() for k in ("steps", "decay_steps", "optimizer", "best_step", "singer", "vibe")}
        extras.update({"lr_entered": self.cv["learning_rate"].get().strip(), "accum": accum, "calc_version": CALC_VERSION,
                       "inst": self.cal_inst.get(), "vocal": self.cal_vocal.get()})
        dataset = self._dataset_name()
        name = self.cal_name.get().strip() or dataset or "run"
        point = cal.make_point(name, dataset, self._backend(), self.genre_var.get(), self._dataset_features(),
                               baseline, worked, extras)
        self.cal_data["points"].append(point)
        cal.save(self.cal_data, self.cal_path)
        self.cal_refresh_view()
        self.cal_tree.selection_set(point["id"])
        note = f"\nSaved '{name}'. Select it and press 'Use selected point as calibration' to make it the datum."
        if best and steps and best < 0.85 * steps:
            note += (f"\nNote: your best checkpoint ({int(best)}) was well before the end of the run ({int(steps)}). Calibration reproduces the "
                     f"Steps/Decay steps you entered ({int(steps)}); enter the run length you want repeated.")
        self.cal_status.set(self.cal_status.get() + note)

    def cal_use_selected(self):
        sel = self.cal_tree.selection()
        if not sel:
            messagebox.showinfo("Calibration", "Select a saved calibration point in the table first.")
            return
        cal.set_active(self.cal_data, sel[0])
        cal.save(self.cal_data, self.cal_path)
        self.use_cal_var.set(True)
        self.cal_refresh_view()
        self.on_options_changed()

    def cal_stop_using(self):
        self.cal_data["active"] = None
        cal.save(self.cal_data, self.cal_path)
        self.cal_refresh_view()
        self.on_options_changed()

    def cal_delete_selected(self):
        sel = self.cal_tree.selection()
        if not sel:
            return
        cal.remove_points(self.cal_data, sel)
        cal.save(self.cal_data, self.cal_path)
        self.cal_refresh_view()
        self.on_options_changed()

    def cal_remove_all(self):
        if not self.cal_data["points"] and not self.cal_data.get("active"):
            return
        if not messagebox.askyesno("Remove ALL calibration",
                                   f"Delete all {len(self.cal_data['points'])} saved calibration point(s) and go back to the "
                                   "plain calculator?\nThis cannot be undone."):
            return
        cal.remove_all(self.cal_data)
        cal.save(self.cal_data, self.cal_path)
        self.cal_refresh_view()
        self.on_options_changed()

    # ------------------------------------------------------------------ calculator actions
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.folder_path.set(folder)

    def on_options_changed(self):
        if self.last_dsp_results:
            self.display_calculated_config()
        else:
            self.cal_refresh_view()

    def copy_calculated_to_adjust_from(self):
        """Load the current recommended settings into the Adjust From boxes, so the feedback below corrects THESE.
        Feedback/peak step are reset so they are not applied twice."""
        c = self.last_config
        if not c:
            messagebox.showinfo("Adjust From", "Analyze a dataset first.")
            return
        for k in ("rank", "alpha", "timing_loss_weight", "steps", "planner_scale", "caption_dropout",
                  "decay_steps", "decay_shape", "accumulation"):
            self.af[k].set(str(c[k]))
        self.af["learning_rate"].set(str(c["learning_rate"]) if self._num(c["learning_rate"]) is not None else "")
        self.inst_feedback_var.set("Good")
        self.vocal_feedback_var.set("Good")
        self.prior_best_step.set("0")
        self.on_options_changed()

    def reset_to_standard(self):
        """Clear the boxes and go back to the plain calculator: no entered settings, no feedback, no calibration."""
        for k, _ in self.af_fields:
            self.af[k].set("")
        self.af["accumulation"].set("1")
        self.inst_feedback_var.set("Good")
        self.vocal_feedback_var.set("Good")
        self.prior_best_step.set("0")
        self.decay_shape_mode_var.set("Auto")
        self.use_cal_var.set(False)           # tick "Apply the active calibration" to switch it back on
        self.on_options_changed()
        if not self.last_dsp_results:
            self.af_status.set("Standard calculation.")

    def run_analysis(self):
        if not check_dependencies_and_warn():
            return

        path = self.folder_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid dataset directory.")
            return

        is_mm3_mode = False   # captions are read from <track>.yue2.txt files only

        self.txt_output.delete("1.0", tk.END)
        self.txt_output.insert(tk.END, f"Scanning dataset: {path}\n" + "-"*65 + "\n")
        self.root.update_idletasks()

        dsp_results = analyze_dataset_dsp(path, is_mm3_mode=is_mm3_mode)

        if not dsp_results:
            messagebox.showwarning("No Audio Found", "No valid audio files could be analyzed.")
            return

        dsp_results["is_mm3"] = is_mm3_mode
        self.last_dsp_results = dsp_results
        self.display_calculated_config()

    def display_calculated_config(self):
        dsp = self.last_dsp_results
        if not dsp:
            return

        accum_val = self._num(self.accum_override.get(), int) or 1
        prior_step_val = self._num(self.prior_best_step.get(), int) or 0
        opt_choice = self.optimizer_choice.get()
        inst_f = self.inst_feedback_var.get()
        voc_f = self.vocal_feedback_var.get()
        shape_mode = self.decay_shape_mode_var.get()

        common = dict(
            num_tracks=dsp['num_parent_tracks'],
            num_chunks=dsp['num_chunks'],
            total_duration_secs=dsp['total_duration_secs'],
            c_std=dsp['c_std'],
            spectral_flux=dsp['spectral_flux'],
            zcr=dsp['zcr'],
            rms_std=dsp['rms_std'],
            avg_caption_words=dsp['avg_caption_words'],
            avg_caption_tags=dsp.get('avg_tags_all'),
            override_accum=accum_val,
            optimizer_choice=opt_choice,
            decay_shape_mode=shape_mode,
        )

        # Plain calculator output (no feedback / calibration / entered settings): what calibration compares against
        self.last_pure = calculate_hotstep_pertinent_settings(
            **common, prior_best_step=0, inst_feedback="Good", vocal_feedback="Good")

        # Settings typed into the Adjust From boxes: they become the base for every setting they cover
        def fnum(key):
            return self._num(self.af[key].get(), float)

        ov = {}
        for key, name, cast in (("rank", "rank", int), ("learning_rate", "lr", float), ("steps", "steps", int),
                                ("accumulation", "accum", int), ("caption_dropout", "dropout", float),
                                ("alpha", "alpha", float), ("timing_loss_weight", "timing", float),
                                ("planner_scale", "planner", float), ("decay_steps", "decay_steps", int)):
            v = fnum(key)
            if v is not None and (v > 0 or key == "caption_dropout"):
                ov[name] = cast(v)
        shape = self.af["decay_shape"].get().strip().capitalize()
        if shape in ("Cosine", "Linear"):
            ov["decay_shape"] = shape
        base_override = ov if any(k in ov for k in ("rank", "lr", "steps", "dropout", "alpha", "timing", "planner", "decay_steps")) else None

        # Active calibration datum (not used while you are running your own settings)
        calib_mult, ap, cal_note = None, self._active_point(), None
        if cal and ap:
            if not self.use_cal_var.get():
                cal_note = f"Calibration '{ap['name']}' is switched off - plain calculator."
            elif base_override:
                cal_note = "Calibration not applied: you are running your own settings from the Adjust From boxes."
            elif ap.get("backend") != self._backend():
                cal_note = f"Calibration '{ap['name']}' is for {ap.get('backend')} - not applied to {self._backend()}."
            else:
                calib_mult = cal.anchor_from_point(ap)

        config = calculate_hotstep_pertinent_settings(
            **common,
            prior_best_step=prior_step_val,
            inst_feedback=inst_f,
            vocal_feedback=voc_f,
            calib_mult=calib_mult,
            base_override=base_override,
        )
        self.last_config = config

        extra = []
        if calib_mult:
            extra.append(f"CALIBRATION DATUM: '{ap['name']}' (dataset: {ap.get('dataset') or '?'}). The calculator is re-anchored on the settings "
                         f"that worked there. Versus the plain calculation: {cal.describe_anchor(calib_mult)}")
        if calib_mult and ap.get("extras", {}).get("calc_version") != CALC_VERSION:
            extra.append("This calibration point was saved before the tag-based caption scaling, so it will not reproduce exactly - re-save it "
                         "(analyze its dataset, enter the settings that worked, Save).")
        if calib_mult and ap.get("extras", {}).get("accum") and int(ap["extras"]["accum"]) != accum_val:
            extra.append(f"The datum run used accumulation {ap['extras']['accum']}; learning rate here is rescaled for your current accumulation {accum_val} (LR / sqrt(accumulation)).")
        if cal_note:
            extra.append(cal_note)

        feedback_given = prior_step_val > 0 or inst_f != "Good" or voc_f not in ("Good", "NA")
        banner = None
        if base_override:
            banner = ("*** YOUR ENTERED SETTINGS (Adjust From boxes), CORRECTED by your feedback ***" if feedback_given
                      else "*** YOUR ENTERED SETTINGS (Adjust From boxes) - no correction applied ***")
        elif calib_mult:
            banner = f"*** CALIBRATED from '{ap['name']}' (dataset: {ap.get('dataset') or '?'}) ***"

        self.txt_output.delete("1.0", tk.END)

        self.txt_output.insert(tk.END, "=== DATASET ANALYSIS DETAILS ===\n")
        if self._dataset_name():
            self.txt_output.insert(tk.END, f"Dataset: {self._dataset_name()}\n")
        self.txt_output.insert(tk.END, f"Parent Tracks: {dsp['num_parent_tracks']}\n")
        self.txt_output.insert(tk.END, f"Total Audio Chunks: {dsp['num_chunks']}\n")
        self.txt_output.insert(tk.END, f"Total Duration: {dsp['total_duration_secs']/60.0:.2f} minutes ({dsp['total_duration_secs']/3600.0:.2f} hours)\n")
        self.txt_output.insert(tk.END, f"Spectral Flux (Transients): {dsp['spectral_flux']:.3f}\n")
        self.txt_output.insert(tk.END, f"Spectral Centroid StdDev: {dsp['c_std']:.2f} Hz\n")
        self.txt_output.insert(tk.END, f"Zero Crossing Rate (ZCR): {dsp['zcr']:.4f}\n")
        self.txt_output.insert(tk.END, f"RMS Energy StdDev: {dsp['rms_std']:.4f}\n")
        self.txt_output.insert(tk.END, f"Captions Found (*.yue2.txt only): {dsp['captioned_tracks']}/{dsp['num_parent_tracks']} tracks ({dsp['caption_ratio']*100:.1f}%)\n")
        if dsp['captioned_tracks'] == 0:
            self.txt_output.insert(tk.END, "  ! No usable <trackname>.yue2.txt files found next to the audio - captions are counted as empty.\n")
        elif dsp.get('yue2_files', dsp['captioned_tracks']) > dsp['captioned_tracks']:
            self.txt_output.insert(tk.END, f"  ! {dsp['yue2_files'] - dsp['captioned_tracks']} .yue2.txt file(s) were empty / had no caption text.\n")
        self.txt_output.insert(tk.END, f"Avg Caption Word Count: {dsp['avg_caption_words']:.1f} words/track\n")
        if dsp.get('avg_caption_tags'):
            self.txt_output.insert(tk.END, f"Avg Caption Tags: {dsp['avg_caption_tags']:.1f} comma-separated tags per captioned track\n")
        if dsp.get('avg_tags_all') is not None:
            self.txt_output.insert(tk.END, f"Caption Richness Factor: {config['caption_density_factor']:.2f} (tags per track averaged over ALL tracks: {dsp['avg_tags_all']:.1f}; {CAPTION_TAGS_FULL} tags = full)\n")
        self.txt_output.insert(tk.END, f"DSP Variance Score: {config['dsp_variance_score']:.2f}\n")
        self.txt_output.insert(tk.END, f"Audio Complexity Factor: {config['audio_complexity']:.2f}\n")
        self.txt_output.insert(tk.END, f"Combined Complexity Score: {config['combined_complexity']:.2f}\n")

        if config['reasons'] or extra:
            self.txt_output.insert(tk.END, "\n--- Calibration & Decay Shape Decision ---\n")
            for r in extra + config['reasons']:
                self.txt_output.insert(tk.END, f" \u2022 {r}\n")

        self.txt_output.insert(tk.END, "="*65 + "\n\n")

        self.txt_output.insert(tk.END, "=== RECOMMENDED HOT-STEP SETTINGS ===\n")
        if banner:
            self.txt_output.insert(tk.END, banner + "\n")
        rows = [
            ("Lora Rank", config['rank']), ("Alpha", config['alpha']),
            ("Timing Loss Weight", config['timing_loss_weight']), ("Steps", config['steps']),
            ("Optimizer", config['optimizer']), ("Learning Rate", config['learning_rate']),
            ("Planner Learning-Rate Scale", config['planner_scale']), ("Caption Dropout", config['caption_dropout']),
            ("Decay Steps", config['decay_steps']), ("Decay Shape", config['decay_shape']),
            ("Accumulation Steps", config['accumulation']),
        ]
        for label, val in rows:
            self.txt_output.insert(tk.END, f"{label}: {val}\n")

        if base_override:
            self.af_status.set("The Recommended Settings are YOUR entered settings" +
                               (", corrected by the feedback below." if feedback_given else " (no feedback given yet).") +
                               " Blank boxes use the calculated value. Use 'Copy calculated settings' to chain the next correction.")
        else:
            self.af_status.set("Standard calculation from the dataset" +
                               (f" + calibration '{ap['name']}'." if calib_mult else "."))

        self.cal_refresh_view()


if __name__ == "__main__":
    root = tk.Tk()
    app = HOTStepCalculatorApp(root)
    root.mainloop()
