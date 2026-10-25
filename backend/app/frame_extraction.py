"""Timestamped, bounded frame extraction using the system ffmpeg binary."""
from pathlib import Path
import re
import subprocess

from .config import VIDEO_FRAME_INTERVAL, VIDEO_MAX_FRAMES, VIDEO_FRAME_MAX_DIMENSION


def extract_frames(video_path: str, output_dir: str, interval: float | None = None,
                   max_frames: int | None = None, max_dimension: int | None = None) -> list[dict]:
    source = Path(video_path)
    if not source.is_file():
        raise ValueError("The uploaded video file is unavailable.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    step = interval or VIDEO_FRAME_INTERVAL
    limit = max_frames or VIDEO_MAX_FRAMES
    size = max_dimension or VIDEO_FRAME_MAX_DIMENSION
    pattern = output / "frame-%04d.jpg"
    vf = f"fps=1/{step},scale=w='min({size},iw)':h='min({size},ih)':force_original_aspect_ratio=decrease"
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
               "-vf", vf, "-frames:v", str(limit), "-q:v", "4", str(pattern)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    frames = sorted(output.glob("frame-*.jpg"))
    if result.returncode != 0 or not frames:
        raise ValueError((result.stderr.strip()[-500:] if result.stderr else "No decodable video frames were found."))
    return [{"frame_number": i, "timestamp_seconds": round((i - 1) * step, 3), "frame_path": str(path)}
            for i, path in enumerate(frames, 1)]


def extract_frames_at(video_path: str, output_dir: str, timestamps: list[float],
                      max_dimension: int | None = None) -> list[dict]:
    """Extract only requested timestamps, recording FFmpeg's actual decoded PTS."""
    source = Path(video_path)
    if not source.is_file():
        raise ValueError("The uploaded video file is unavailable.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    size = max_dimension or VIDEO_FRAME_MAX_DIMENSION
    results = []
    seen = set()
    for ordinal, requested in enumerate(sorted(timestamps), 1):
        requested = max(0.0, float(requested))
        key = round(requested, 3)
        if key in seen:
            continue
        seen.add(key)
        frame_path = output / f"frame-{ordinal:06d}.jpg"
        vf = f"scale=w='min({size},iw)':h='min({size},ih)':force_original_aspect_ratio=decrease,showinfo"
        # A target at/after the final presentation timestamp can yield no frame.
        # Retry slightly earlier only in that case; persist FFmpeg's actual PTS.
        seek_points = [requested]
        seek_points.extend(max(0.0, requested - offset) for offset in (0.05, 0.15, 0.35, 0.75))
        completed = None
        for seek_point in dict.fromkeys(seek_points):
            command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-copyts", "-ss", f"{seek_point:.3f}",
                       "-i", str(source), "-frames:v", "1", "-vf", vf, "-q:v", "4", str(frame_path)]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeError(f"FFmpeg could not extract the frame requested at {requested:.2f}s.") from exc
            if completed.returncode == 0 and frame_path.is_file() and frame_path.stat().st_size:
                break
        if completed is None or completed.returncode or not frame_path.is_file() or not frame_path.stat().st_size:
            detail = completed.stderr[-300:] if completed is not None and completed.stderr else "no frame was decoded"
            raise RuntimeError(f"FFmpeg could not extract the frame requested at {requested:.2f}s: {detail}")
        pts_match = re.search(r"pts_time:([0-9]+(?:\.[0-9]+)?)", completed.stderr)
        actual = float(pts_match.group(1)) if pts_match else requested
        results.append({"frame_number": ordinal, "timestamp_seconds": actual,
                        "requested_timestamp_seconds": requested, "frame_path": str(frame_path)})
    return results
