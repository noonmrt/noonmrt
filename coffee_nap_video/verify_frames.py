#!/usr/bin/env python3
"""Extracts key frames from the rendered MP4 and builds a 4x2 QA contact sheet."""

import subprocess
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent
MP4_PATH = BASE_DIR / "output" / "coffee_nap_3d_poligonos_80s.mp4"
QA_DIR = BASE_DIR / "build" / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

# Pick timestamps inside key scenes (mid-speech & mid-HUD animation)
STAMPS = [2.2, 17.5, 22.8, 32.8, 57.5, 62.5, 72.5, 77.2]


def main():
    thumbs = []
    for idx, ts in enumerate(STAMPS):
        out_jpg = QA_DIR / f"frame_{idx + 1:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{ts:.2f}",
                "-i", str(MP4_PATH),
                "-frames:v", "1",
                str(out_jpg),
            ],
            check=True,
        )
        im = Image.open(out_jpg).resize((360, 640), Image.Resampling.LANCZOS)
        thumbs.append(im)

    cols, rows = 4, 2
    sheet = Image.new("RGB", (cols * 360, rows * 640), (10, 12, 20))
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * 360, r * 640))

    sheet_path = QA_DIR / "qa_contact_sheet.jpg"
    sheet.save(sheet_path, quality=92)
    print(f"Saved QA contact sheet -> {sheet_path}")


if __name__ == "__main__":
    main()
