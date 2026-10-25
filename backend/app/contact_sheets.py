"""Small labeled contact sheets using FFmpeg and NumPy (no image-library dependency)."""
from pathlib import Path
import subprocess

import numpy as np


_FONT = {
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    " ": ("00000",) * 7,
}


def _thumbnail(path: str, width: int, height: int) -> np.ndarray:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-frames:v", "1",
               "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    result = subprocess.run(command, capture_output=True, timeout=20)
    expected = width * height * 3
    if result.returncode or len(result.stdout) != expected:
        raise ValueError(f"Could not decode contact-sheet image: {Path(path).name}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def _draw_label(canvas: np.ndarray, x: int, y: int, label: str):
    scale = 2
    for char in label:
        glyph = _FONT.get(char, _FONT[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    canvas[y + row * scale:y + (row + 1) * scale,
                           x + col * scale:x + (col + 1) * scale] = 255
        x += 6 * scale


def create_contact_sheet(frames: list[dict], output_path: str, columns: int = 3,
                         thumb_width: int = 320, thumb_height: int = 180) -> str:
    if not frames:
        raise ValueError("At least one frame is required for a contact sheet.")
    if any(not Path(frame["frame_path"]).is_file() for frame in frames):
        raise ValueError("A contact-sheet frame is missing.")
    columns = max(1, min(columns, len(frames)))
    label_height, gutter, margin = 24, 8, 8
    cell_height = label_height + thumb_height
    rows = (len(frames) + columns - 1) // columns
    width = margin * 2 + columns * thumb_width + (columns - 1) * gutter
    height = margin * 2 + rows * cell_height + (rows - 1) * gutter
    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, col = divmod(index, columns)
        x, y = margin + col * (thumb_width + gutter), margin + row * (cell_height + gutter)
        sheet[y:y + label_height, x:x + thumb_width] = 24
        _draw_label(sheet, x + 6, y + 3, f"F{int(frame['frame_id'])}")
        sheet[y + label_height:y + cell_height, x:x + thumb_width] = _thumbnail(frame["frame_path"], thumb_width, thumb_height)
    ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + sheet.tobytes()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-vcodec", "ppm",
               "-i", "pipe:0", "-frames:v", "1", "-q:v", "3", str(destination)]
    result = subprocess.run(command, input=ppm, capture_output=True, timeout=30)
    if result.returncode or not destination.is_file():
        raise RuntimeError("FFmpeg could not encode the contact sheet.")
    return str(destination)
