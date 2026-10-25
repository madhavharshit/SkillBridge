"""Cheap streaming video scan and explainable candidate-window selection."""
from fractions import Fraction
import json
import re
import subprocess
import tempfile

import numpy as np


def probe_video(path: str) -> dict:
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        result.check_returncode()
        metadata = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not inspect video metadata with ffprobe.") from exc
    stream = next((item for item in metadata.get("streams", []) if item.get("codec_type") == "video"), None)
    if not stream:
        raise ValueError("The selected file does not contain a video stream.")
    try:
        rate = float(Fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"))
    except (ValueError, ZeroDivisionError):
        rate = 0.0
    duration = float(stream.get("duration") or metadata.get("format", {}).get("duration") or 0)
    tags = stream.get("tags", {})
    side_data = stream.get("side_data_list", [])
    rotation = next((int(item.get("rotation", 0)) for item in side_data if "rotation" in item), None)
    return {
        "duration_seconds": duration,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "frame_rate": rate,
        "frame_count_estimate": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "codec": stream.get("codec_name", "unknown"),
        "time_base": stream.get("time_base"),
        "rotation_degrees": rotation if rotation is not None else int(tags.get("rotate", 0) or 0),
    }


def adaptive_motion_scan_fps(duration: float, source_frame_rate: float,
                             maximum_fps: float, target_samples: int) -> float:
    """Use metadata and a sample ceiling to bound inexpensive coarse scanning."""
    maximum_fps = max(0.1, float(maximum_fps))
    target_samples = max(1, int(target_samples))
    if duration <= 0:
        return min(maximum_fps, source_frame_rate) if source_frame_rate > 0 else maximum_fps
    duration_limited_rate = float(target_samples) / float(duration)
    rate = min(maximum_fps, duration_limited_rate)
    if source_frame_rate > 0:
        rate = min(rate, float(source_frame_rate))
    return max(1e-6, rate)


def adaptive_candidate_baseline_interval(duration: float, requested_interval: float,
                                         maximum_baseline_candidates: int) -> float:
    """Keep uniform candidate rows bounded as clip duration grows."""
    requested_interval = max(0.1, float(requested_interval))
    maximum_baseline_candidates = max(1, int(maximum_baseline_candidates))
    if duration <= 0:
        return requested_interval
    # ``build_candidates`` includes the initial timestamp and both ends of its
    # inclusive loop. Leave half an interval of headroom so that rounding cannot
    # produce one extra baseline row over the configured ceiling.
    duration_limited_interval = float(duration) / max(0.5, maximum_baseline_candidates - 0.5)
    return max(requested_interval, duration_limited_interval)


def _read_exact(stream, length: int) -> bytes:
    parts = []
    remaining = length
    while remaining:
        block = stream.read(remaining)
        if not block:
            break
        parts.append(block)
        remaining -= len(block)
    return b"".join(parts)


def scan_motion(path: str, duration: float, fps: float = 5.0,
                width: int = 160, height: int = 90,
                local_threshold: float = 0.035) -> dict:
    """Stream low-resolution grayscale frames; retain only scoring metadata/peaks."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-i", path,
               "-vf", f"fps={fps}:start_time=0,scale={width}:{height}:flags=area,format=gray,showinfo",
               "-an", "-sn", "-dn", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    frame_bytes = width * height
    samples = []
    # showinfo's per-frame PTS output can exceed a pipe buffer on long videos. Spool
    # it to a temporary file rather than buffering it in memory or blocking FFmpeg.
    with tempfile.TemporaryFile(mode="w+b") as diagnostic_stream:
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=diagnostic_stream)
        except OSError as exc:
            raise RuntimeError("FFmpeg is required for the low-resolution motion scan.") from exc
        previous = None
        index = 0
        try:
            assert process.stdout is not None
            while True:
                raw = _read_exact(process.stdout, frame_bytes)
                if len(raw) != frame_bytes:
                    break
                gray = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
                if previous is not None:
                    diff = np.abs(gray.astype(np.int16) - previous.astype(np.int16))
                    global_change = float(diff.mean() / 255.0)
                    changed_fraction = float(np.mean(diff > 18))
                    # The maximum tile score preserves small hand/tool movement against a static background.
                    tile_scores = [float(tile.mean() / 255.0) for row in np.array_split(diff, 9, axis=0)
                                   for tile in np.array_split(row, 16, axis=1)]
                    local_change = max(tile_scores, default=0.0)
                    scene_change = changed_fraction >= 0.48 and global_change >= 0.08
                    score = min(1.0, 0.45 * global_change + 0.55 * local_change)
                    samples.append({"global_change": global_change,
                                    "localized_change": local_change, "changed_fraction": changed_fraction,
                                    "scene_change": scene_change, "motion_score": score})
                previous = gray
                index += 1
        except Exception:
            process.kill()
            raise
        finally:
            if process.stdout:
                process.stdout.close()
        return_code = process.wait()
        if return_code:
            diagnostic_stream.seek(0, 2)
            size = diagnostic_stream.tell()
            diagnostic_stream.seek(max(0, size - 400))
            stderr = diagnostic_stream.read().decode("utf-8", "replace")
            raise RuntimeError(f"FFmpeg motion scan failed: {stderr[-400:] or 'no frames decoded'}")
        diagnostic_stream.seek(0)
        timestamp_rows = re.findall(rb"pts_time:([-+]?(?:[0-9]*[.])?[0-9]+)", diagnostic_stream.read())
        timestamps = [float(item) for item in timestamp_rows]
        if len(timestamps) != index:
            raise RuntimeError("FFmpeg motion scan could not align sampled frames with presentation timestamps.")
        for sample_index, sample in enumerate(samples, start=1):
            sample["timestamp_seconds"] = round(timestamps[sample_index], 3)
    # Local maxima suppress repeated neighboring samples without dropping low-motion baseline coverage.
    peaks = []
    for i, row in enumerate(samples):
        if row["localized_change"] < local_threshold and not row["scene_change"]:
            continue
        left = samples[max(0, i - 2)]["motion_score"]
        right = samples[min(len(samples) - 1, i + 2)]["motion_score"]
        if row["motion_score"] >= left and row["motion_score"] >= right:
            peaks.append(row)
    # Keep local activity and scene cuts separate; collapse peaks closer than 0.8 seconds.
    selected = []
    for row in sorted(peaks, key=lambda item: (item["timestamp_seconds"], -item["motion_score"])):
        if selected and row["timestamp_seconds"] - selected[-1]["timestamp_seconds"] < 0.8:
            if row["motion_score"] > selected[-1]["motion_score"]:
                selected[-1] = row
        else:
            selected.append(row)
    return {"sample_count": index, "sample_interval_seconds": round(1 / fps, 4), "peaks": selected}


_CUES = {
    "en": re.compile(r"\b(start|stop|adjust|position|check|inspect|measure|test|confirm|correct|fault|caution|exception|failure|temperature|pressure|tighten|loosen)\b", re.I),
    "ja": re.compile(r"開始|停止|調整|位置|確認|点検|測定|検査|テスト|修正|異常|注意|例外|失敗|温度|圧力|締め|緩め"),
}


def build_candidates(duration: float, motion_peaks: list[dict], transcript: list[dict],
                     baseline_interval: float = 2.0) -> list[dict]:
    candidates: dict[float, dict] = {}

    def add(ts: float, signal: str, score: float, parts: dict | None = None, reason: str | None = None):
        ts = round(min(max(ts, 0.0), max(duration, 0.0)), 2)
        row = candidates.setdefault(ts, {"timestamp_seconds": ts, "score": 0.0,
                                         "score_components": {}, "reasons": []})
        row["score_components"][signal] = max(score, row["score_components"].get(signal, 0.0))
        row["score"] = max(row["score"], score)
        if signal not in row["reasons"]:
            row["reasons"].append(signal if not reason else reason)
        if parts:
            row["score_components"].update(parts)

    ts = 0.0
    while ts <= duration:
        add(ts, "uniform_baseline", 0.15)
        ts += baseline_interval
    for peak in motion_peaks:
        add(peak["timestamp_seconds"], "scene_boundary" if peak["scene_change"] else "localized_motion",
            min(0.95, 0.3 + peak["motion_score"]),
            {key: peak[key] for key in ("global_change", "localized_change", "changed_fraction", "motion_score")})
    for segment in transcript:
        text = segment.get("text", "")
        language = "ja" if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text) else "en"
        cues = bool(_CUES[language].search(text))
        start = float(segment.get("start_seconds", 0) or 0)
        end = float(segment.get("end_seconds", start) or start)
        add(start, "transcript_boundary", 0.55, reason="transcript boundary")
        if cues:
            add((start + end) / 2, "transcript_cue", 0.8, reason="procedural transcript cue")
    return sorted(candidates.values(), key=lambda row: row["timestamp_seconds"])


def build_windows(candidates: list[dict], duration: float, radius: float = 2.0,
                  merge_gap: float = 1.0, max_duration: float = 20.0,
                  baseline_coverage_interval_seconds: float = 30.0) -> list[dict]:
    baseline_candidates = [row for row in candidates if "uniform_baseline" in row["score_components"]]

    def sparse_baseline_rows():
        if not baseline_candidates:
            return []
        timestamps = [float(row["timestamp_seconds"]) for row in baseline_candidates]
        differences = np.diff(timestamps)
        baseline_step = float(np.median(differences[differences > 0])) if np.any(differences > 0) else baseline_coverage_interval_seconds
        stride = max(1, int(round(baseline_coverage_interval_seconds / max(0.1, baseline_step))))
        selected = baseline_candidates[::stride]
        if selected[-1] is not baseline_candidates[-1]:
            selected.append(baseline_candidates[-1])
        return selected

    seeds = [row for row in candidates if "localized_motion" in row["score_components"]
             or "scene_boundary" in row["score_components"] or "transcript_cue" in row["score_components"]]
    if not seeds:
        seeds = sparse_baseline_rows()
    else:
        # Sparse baseline anchors protect static control panels and other low-motion evidence.
        seeds.extend(sparse_baseline_rows())
    seeds = sorted({row["timestamp_seconds"]: row for row in seeds}.values(), key=lambda row: row["timestamp_seconds"])
    groups: list[dict] = []
    for seed in seeds:
        start = max(0.0, seed["timestamp_seconds"] - radius)
        end = min(duration, seed["timestamp_seconds"] + radius)
        if groups and start <= groups[-1]["end_seconds"] + merge_gap and end - groups[-1]["start_seconds"] <= max_duration:
            group = groups[-1]
            group["end_seconds"] = end
            group["candidate_timestamps"].append(seed["timestamp_seconds"])
            group["score"] = max(group["score"], seed["score"])
            group["reasons"] = sorted(set(group["reasons"] + seed["reasons"]))
        else:
            groups.append({"start_seconds": start, "end_seconds": end,
                           "candidate_timestamps": [seed["timestamp_seconds"]],
                           "score": seed["score"], "reasons": list(seed["reasons"])})
    return groups


def adaptive_contact_budget(duration: float, candidates: list[dict], maximum: int,
                            coverage_interval_seconds: float = 30.0) -> int:
    """Choose a screening budget from duration and the number of actionable signals."""
    if not candidates or maximum < 1:
        return 0
    baseline = max(1, int(np.ceil(max(0.0, duration) / max(1.0, coverage_interval_seconds))))
    priority_count = sum(bool(row.get("priority")) for row in candidates)
    target = max(min(6, maximum), baseline + min(priority_count, baseline * 2))
    return min(len(candidates), maximum, target)


def select_contact_candidates(candidates: list[dict], limit: int = 27) -> list[dict]:
    if len(candidates) <= limit:
        return candidates
    priority = [row for row in candidates if "localized_motion" in row["score_components"]
                or "scene_boundary" in row["score_components"] or "transcript_cue" in row["score_components"]]
    selected = sorted(priority, key=lambda row: row["score"], reverse=True)[: max(1, limit * 2 // 3)]
    selected_times = {row["timestamp_seconds"] for row in selected}
    remaining = [row for row in candidates if row["timestamp_seconds"] not in selected_times]
    slots = limit - len(selected)
    if slots > 0 and remaining:
        indices = np.linspace(0, len(remaining) - 1, min(slots, len(remaining)), dtype=int)
        selected.extend(remaining[index] for index in sorted(set(indices.tolist())))
    return sorted(selected, key=lambda row: row["timestamp_seconds"])


def adaptive_detail_budget(duration: float, windows: list[dict], maximum: int,
                           coverage_interval_seconds: float = 30.0) -> int:
    """Reserve broad timeline coverage, then scale detail capacity with action windows."""
    if not windows or maximum < 1:
        return 0
    priority_count = sum(bool(window.get("priority")) for window in windows)
    baseline_target = max(3, int(np.ceil(max(0.0, duration) / max(1.0, coverage_interval_seconds))))
    baseline_slots = min(maximum, baseline_target)
    if priority_count:
        baseline_slots = min(baseline_slots, max(1, maximum // 3))
    minimum = min(maximum, 3)
    return min(maximum, max(minimum, baseline_slots + min(priority_count * 2, maximum - baseline_slots)))


def select_dense_timestamps(windows: list[dict], budget: int = 25, fps: float = 3.0,
                            duration: float | None = None,
                            coverage_interval_seconds: float = 30.0) -> dict[int, list[float]]:
    """Select a deterministic timeline plan within budget.

    Baseline samples are distributed over the clip first. Remaining capacity goes to
    high-scoring action windows. One center sample is retained per chosen window
    before any window receives additional before/after detail frames.
    """
    if not windows or budget < 1:
        return {}
    duration = max(float(duration or max(row["end_seconds"] for row in windows)), 0.0)
    ordered = sorted(enumerate(windows), key=lambda item: (item[1]["start_seconds"], item[0]))
    selections: dict[int, list[float]] = {}
    used: set[float] = set()

    def add(index: int, timestamp: float) -> bool:
        timestamp = round(min(max(float(timestamp), 0.0), duration), 3)
        if timestamp in used or len(used) >= budget:
            return False
        used.add(timestamp)
        selections.setdefault(index, []).append(timestamp)
        return True

    # Choose baseline windows spread across the actual clip, not the first N timestamps.
    baseline_rows = [(index, window) for index, window in ordered if window.get("baseline")]
    if not baseline_rows:
        baseline_rows = ordered
    has_priority = any(window.get("priority") for window in windows)
    baseline_limit = max(1, budget // 3) if has_priority else budget
    coverage_target = max(1, int(np.ceil(duration / max(1.0, coverage_interval_seconds))))
    baseline_target = min(len(baseline_rows), baseline_limit,
                          coverage_target if has_priority else budget)
    targets = ([duration / 2] if baseline_target == 1 else
               np.linspace(0.0, duration, baseline_target).tolist())
    chosen_baseline: set[int] = set()
    for target in targets:
        available = [(index, window) for index, window in baseline_rows if index not in chosen_baseline]
        if not available:
            break
        index, window = min(available, key=lambda pair: (
            abs((pair[1]["start_seconds"] + pair[1]["end_seconds"]) / 2 - target), pair[1]["start_seconds"]))
        center = (window["start_seconds"] + window["end_seconds"]) / 2
        if add(index, center):
            chosen_baseline.add(index)

    ranked = sorted(ordered, key=lambda item: (
        bool(item[1].get("priority")), float(item[1].get("score", 0)),
        -float(item[1]["start_seconds"])), reverse=True)
    selected_indices = set(selections)
    for index, window in ranked:
        if len(used) >= budget:
            break
        if not window.get("priority") and any(item.get("priority") for item in windows):
            continue
        center = (window["start_seconds"] + window["end_seconds"]) / 2
        if add(index, center):
            selected_indices.add(index)

    # Keep chronology inside top-ranked windows where there is capacity.
    for index, window in ranked:
        if len(used) >= budget:
            break
        if index not in selected_indices:
            continue
        add(index, window["start_seconds"])
        if len(used) < budget:
            add(index, window["end_seconds"])

    return {index: sorted(set(times)) for index, times in selections.items()}


def plan_detail_frames(windows: list[dict], duration: float, maximum: int,
                       coverage_interval_seconds: float = 30.0) -> dict:
    budget = adaptive_detail_budget(duration, windows, maximum, coverage_interval_seconds)
    timestamps = select_dense_timestamps(windows, budget, duration=duration,
                                         coverage_interval_seconds=coverage_interval_seconds)
    selected = set(timestamps)
    priorities = [float(window.get("score", 0)) for window in windows if window.get("priority")]
    threshold = float(np.quantile(priorities, 0.75)) if priorities else None
    unresolved = [index for index, window in enumerate(windows)
                  if window.get("priority") and index not in selected and threshold is not None
                  and float(window.get("score", 0)) >= threshold]
    return {"budget": budget, "timestamps_by_window": timestamps,
            "selected_window_indices": sorted(selected),
            "unresolved_important_indices": unresolved,
            "screened_out_indices": [index for index in range(len(windows))
                                     if index not in selected and index not in unresolved],
            "baseline_interval_seconds": coverage_interval_seconds}
