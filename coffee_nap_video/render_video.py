#!/usr/bin/env python3
"""
High-FPS 1080x1920 (9:16) 3D Polygon Video Compositor & Renderer.
Renders 16 ultra-detailed 3D low-poly scenes (alternating every 5.0s = 80.0s exact)
with smooth gimbal camera motion, floating faceted geometric particles,
animated scientific HUD overlays, word-synced kinetic PT-BR subtitles,
5-second transitions, and mastered stereo audio.
"""

import json
import math
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent
SCENES_DIR = BASE_DIR / "assets" / "scenes"
AUDIO_PATH = BASE_DIR / "audio" / "master_audio.wav"
TIMELINE_PATH = BASE_DIR / "build" / "timeline.json"
SEGMENTS_DIR = BASE_DIR / "build" / "segments"
OUTPUT_DIR = BASE_DIR / "output"
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1920
FPS = 30
SCENE_DUR = 5.0
FRAMES_PER_SCENE = int(SCENE_DUR * FPS)  # 150 frames
TRANSITION_FRAMES = 6  # 0.20s smooth crossfade on scene boundary

FONT_DIR = Path("/usr/share/fonts/opentype/inter")
FONT_BLACK = str(FONT_DIR / "InterDisplay-Black.otf")
FONT_EXTRABOLD = str(FONT_DIR / "InterDisplay-ExtraBold.otf")
FONT_BOLD = str(FONT_DIR / "InterDisplay-Bold.otf")
FONT_SEMIBOLD = str(FONT_DIR / "InterDisplay-SemiBold.otf")


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def load_prescaled_image(img_name: str) -> Image.Image:
    """Loads scene image and upscales cleanly once with Lanczos to 1296x2304 (1.2x overscan)."""
    im = Image.open(SCENES_DIR / img_name).convert("RGB")
    target_w = int(W * 1.24)  # 1339
    target_h = int(H * 1.24)  # 2380
    # Center-crop to exact 9:16 aspect ratio before resizing
    iw, ih = im.size
    target_ar = W / H
    cur_ar = iw / ih
    if cur_ar > target_ar:
        new_w = int(ih * target_ar)
        left = (iw - new_w) // 2
        im = im.crop((left, 0, left + new_w, ih))
    else:
        new_h = int(iw / target_ar)
        top = (ih - new_h) // 2
        im = im.crop((0, top, iw, top + new_h))
    return im.resize((target_w, target_h), Image.Resampling.LANCZOS)


def sample_camera_frame(base_img: Image.Image, progress: float, motion: str) -> Image.Image:
    """
    Smooth sub-pixel gimbal camera motion using PIL affine transform.
    progress in [0, 1] over 5.0 seconds.
    """
    bw, bh = base_img.size
    p = progress
    # Gentle floating gimbal sway
    sway_x = math.sin(p * math.pi * 1.5) * 8.0
    sway_y = math.cos(p * math.pi * 1.2) * 6.0

    if motion == "zoom_in_center":
        scale = 1.03 + 0.12 * p
        cx = bw * 0.5 + sway_x
        cy = bh * 0.48 + sway_y
    elif motion == "zoom_in_tight":
        scale = 1.04 + 0.16 * p
        cx = bw * 0.5 + sway_x * 0.7
        cy = bh * 0.47 + sway_y * 0.7
    elif motion == "zoom_out_center":
        scale = 1.17 - 0.12 * p
        cx = bw * 0.5 + sway_x
        cy = bh * 0.49 + sway_y
    elif motion == "pan_right_zoom":
        scale = 1.09 + 0.08 * p
        cx = bw * (0.465 + 0.07 * p)
        cy = bh * 0.49 + sway_y
    elif motion == "pan_up_zoom":
        scale = 1.09 + 0.08 * p
        cx = bw * 0.5 + sway_x
        cy = bh * (0.54 - 0.09 * p)
    elif motion == "pan_down_zoom":
        scale = 1.09 + 0.08 * p
        cx = bw * 0.5 + sway_x
        cy = bh * (0.45 + 0.09 * p)
    else:
        scale = 1.05 + 0.10 * p
        cx = bw * 0.5
        cy = bh * 0.5

    # Viewport size in base_img coordinates
    vw = bw / scale
    vh = bh / scale
    x0 = max(0.0, min(bw - vw, cx - vw * 0.5))
    y0 = max(0.0, min(bh - vh, cy - vh * 0.5))

    # Affine coefficients mapping output (0..W, 0..H) -> input (x0..x0+vw, y0..y0+vh)
    sx = vw / W
    sy = vh / H
    return base_img.transform(
        (W, H),
        Image.Transform.AFFINE,
        (sx, 0.0, x0, 0.0, sy, y0),
        resample=Image.Resampling.BILINEAR,
    )


def build_scrim_lut() -> tuple[np.ndarray, np.ndarray]:
    """Precomputes subtle cinematic top + bottom dark gradient multipliers for fast numpy compositing."""
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    # Top subtle darkening for HUD legibility (top 18%)
    top_dark = np.clip((0.18 - y) / 0.18, 0.0, 1.0) ** 1.6 * 0.48
    # Bottom smooth scrim for captions (bottom 30%)
    bot_dark = np.clip((y - 0.68) / 0.32, 0.0, 1.0) ** 1.45 * 0.78
    alpha = np.clip(top_dark + bot_dark, 0.0, 0.82)
    mult = (1.0 - alpha)[:, None, None]  # (H, 1, 1)
    return mult, alpha


def build_particles(scene_id: int, count: int = 20) -> list[dict]:
    rng = np.random.default_rng(1000 + scene_id * 37)
    pts = []
    for _ in range(count):
        pts.append({
            "x": float(rng.uniform(40, W - 40)),
            "y": float(rng.uniform(140, H - 220)),
            "vy": float(rng.uniform(-48, -16)),
            "vx": float(rng.uniform(-14, 14)),
            "size": float(rng.uniform(7, 20)),
            "sides": int(rng.choice([3, 4, 6])),
            "rot": float(rng.uniform(0, math.tau)),
            "vrot": float(rng.uniform(-1.2, 1.2)),
            "alpha": int(rng.uniform(35, 95)),
        })
    return pts


def draw_particles(overlay: Image.Image, particles: list[dict], local_t: float, accent_rgb: tuple[int, int, int]):
    draw = ImageDraw.Draw(overlay, "RGBA")
    for pt in particles:
        px = (pt["x"] + pt["vx"] * local_t) % W
        py = (pt["y"] + pt["vy"] * local_t) % H
        r = pt["size"]
        rot = pt["rot"] + pt["vrot"] * local_t
        sides = pt["sides"]
        poly = [
            (px + r * math.cos(rot + math.tau * k / sides),
             py + r * math.sin(rot + math.tau * k / sides))
            for k in range(sides)
        ]
        # Subtle faceted polygon fill + crisp bright edge
        fill_col = (accent_rgb[0], accent_rgb[1], accent_rgb[2], int(pt["alpha"] * 0.45))
        edge_col = (255, 255, 255, pt["alpha"])
        draw.polygon(poly, fill=fill_col, outline=edge_col)


def draw_progress_bar(draw: ImageDraw.ImageDraw, scene_idx: int, total_scenes: int, progress: float, accent_rgb: tuple[int, int, int]):
    margin_x = 36
    gap = 7
    bar_y = 44
    bar_h = 7
    total_w = W - 2 * margin_x
    seg_w = (total_w - gap * (total_scenes - 1)) / total_scenes

    for i in range(total_scenes):
        x0 = margin_x + i * (seg_w + gap)
        x1 = x0 + seg_w
        if i < scene_idx:
            draw.rounded_rectangle([x0, bar_y, x1, bar_y + bar_h], radius=3, fill=(255, 255, 255, 215))
        elif i == scene_idx:
            draw.rounded_rectangle([x0, bar_y, x1, bar_y + bar_h], radius=3, fill=(255, 255, 255, 65))
            fill_x1 = x0 + max(4.0, seg_w * progress)
            draw.rounded_rectangle([x0, bar_y, fill_x1, bar_y + bar_h], radius=3, fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 255))
        else:
            draw.rounded_rectangle([x0, bar_y, x1, bar_y + bar_h], radius=3, fill=(255, 255, 255, 55))


def draw_hud(draw: ImageDraw.ImageDraw, scene: dict, local_t: float, fonts: dict):
    """Draws sleek top scientific HUD pill badge + animated data telemetry where relevant."""
    accent_rgb = hex_to_rgb(scene["badge_color"])
    enter = ease_out_cubic(min(1.0, local_t / 0.42))
    slide_y = int((1.0 - enter) * -28)
    alpha_scale = enter

    badge_text = scene["badge"]
    f_badge = fonts["badge"]
    bbox = draw.textbbox((0, 0), badge_text, font=f_badge)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad_x = 32
    icon_space = 30
    pill_w = tw + pad_x * 2 + icon_space
    pill_h = 64
    px0 = (W - pill_w) // 2
    py0 = 92 + slide_y

    # Glassmorphic dark pill + crisp accent border
    draw.rounded_rectangle(
        [px0, py0, px0 + pill_w, py0 + pill_h],
        radius=32,
        fill=(10, 14, 26, int(195 * alpha_scale)),
        outline=(accent_rgb[0], accent_rgb[1], accent_rgb[2], int(235 * alpha_scale)),
        width=2,
    )

    # Rotating/pulsing geometric diamond marker
    cx = px0 + pad_x + 8
    cy = py0 + pill_h // 2
    pulse = 1.0 + 0.18 * math.sin(local_t * 5.0)
    dr = 9.0 * pulse
    diamond = [(cx, cy - dr), (cx + dr, cy), (cx, cy + dr), (cx - dr, cy)]
    draw.polygon(diamond, fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], int(255 * alpha_scale)))

    # Badge label
    draw.text(
        (px0 + pad_x + icon_space, py0 + (pill_h - th) // 2 - 4),
        badge_text,
        font=f_badge,
        fill=(248, 250, 252, int(255 * alpha_scale)),
    )

    # Extra animated data modules on key scientific scenes
    hud_type = scene.get("hud_type", "badge")
    card_enter = ease_out_cubic(max(0.0, min(1.0, (local_t - 0.25) / 0.55)))
    if card_enter <= 0.01:
        return

    if hud_type == "nasa_bars":
        # Animated NASA +34% performance / +54% alertness telemetry card
        cw, ch = 820, 198
        cx0 = (W - cw) // 2
        cy0 = 180 + int((1.0 - card_enter) * 20)
        draw.rounded_rectangle(
            [cx0, cy0, cx0 + cw, cy0 + ch],
            radius=24,
            fill=(10, 16, 30, int(190 * card_enter)),
            outline=(16, 185, 129, int(150 * card_enter)),
            width=2,
        )
        bar_anim = ease_out_cubic(max(0.0, min(1.0, (local_t - 0.55) / 1.4)))
        items = [
            ("DESEMPENHO", 34, 0.56, (16, 185, 129)),
            ("ESTADO DE ALERTA", 54, 0.86, (6, 182, 212)),
        ]
        for idx, (label, val, max_ratio, col) in enumerate(items):
            iy = cy0 + 28 + idx * 82
            cur_val = int(round(val * bar_anim))
            draw.text((cx0 + 32, iy), label, font=fonts["hud_sm"], fill=(203, 213, 225, int(255 * card_enter)))
            val_str = f"+{cur_val}%"
            vb = draw.textbbox((0, 0), val_str, font=fonts["hud_val"])
            draw.text((cx0 + cw - 32 - (vb[2] - vb[0]), iy - 6), val_str, font=fonts["hud_val"], fill=(col[0], col[1], col[2], int(255 * card_enter)))
            bx0, bx1, by0, by1 = cx0 + 32, cx0 + cw - 32, iy + 36, iy + 54
            draw.rounded_rectangle([bx0, by0, bx1, by1], radius=9, fill=(255, 255, 255, int(35 * card_enter)))
            fw = max(18, int((bx1 - bx0) * max_ratio * bar_anim))
            draw.rounded_rectangle([bx0, by0, bx0 + fw, by1], radius=9, fill=(col[0], col[1], col[2], int(245 * card_enter)))

    elif hud_type == "cliff_gauge":
        cw, ch = 820, 134
        cx0 = (W - cw) // 2
        cy0 = 180 + int((1.0 - card_enter) * 20)
        draw.rounded_rectangle(
            [cx0, cy0, cx0 + cw, cy0 + ch],
            radius=24,
            fill=(10, 16, 30, int(190 * card_enter)),
            outline=(239, 68, 68, int(150 * card_enter)),
            width=2,
        )
        bx0, bx1, by0, by1 = cx0 + 36, cx0 + cw - 36, cy0 + 58, cy0 + 80
        mid = bx0 + int((bx1 - bx0) * 0.56)
        draw.rounded_rectangle([bx0, by0, mid - 4, by1], radius=10, fill=(16, 185, 129, int(225 * card_enter)))
        draw.rounded_rectangle([mid + 4, by0, bx1, by1], radius=10, fill=(239, 68, 68, int(225 * card_enter)))
        draw.text((bx0, cy0 + 20), "0–26 MIN • PICO", font=fonts["hud_sm"], fill=(16, 185, 129, int(255 * card_enter)))
        rb = draw.textbbox((0, 0), "30+ MIN • QUEDA", font=fonts["hud_sm"])
        draw.text((bx1 - (rb[2] - rb[0]), cy0 + 20), "30+ MIN • QUEDA", font=fonts["hud_sm"], fill=(248, 113, 113, int(255 * card_enter)))
        # Sliding marker crossing the 30 min threshold
        slide = ease_in_out(max(0.0, min(1.0, (local_t - 0.5) / 2.8)))
        mx = bx0 + int((bx1 - bx0) * (0.22 + 0.65 * slide))
        draw.ellipse([mx - 14, by0 - 3, mx + 14, by1 + 3], fill=(255, 255, 255, int(255 * card_enter)), outline=(15, 23, 42, 255), width=3)
        draw.text((mid - 38, by1 + 8), "30 MIN", font=fonts["hud_xs"], fill=(248, 250, 252, int(210 * card_enter)))

    elif hud_type == "pressure_meter":
        cw, ch = 760, 124
        cx0 = (W - cw) // 2
        cy0 = 180 + int((1.0 - card_enter) * 20)
        draw.rounded_rectangle(
            [cx0, cy0, cx0 + cw, cy0 + ch],
            radius=24,
            fill=(10, 16, 30, int(190 * card_enter)),
            outline=(249, 115, 22, int(155 * card_enter)),
            width=2,
        )
        fill_p = 0.28 + 0.64 * ease_out_cubic(max(0.0, min(1.0, (local_t - 0.3) / 3.6)))
        pct = int(round(fill_p * 100))
        draw.text((cx0 + 32, cy0 + 22), "PRESSÃO DE SONO (ADENOSINA)", font=fonts["hud_sm"], fill=(226, 232, 240, int(255 * card_enter)))
        pb = draw.textbbox((0, 0), f"{pct}%", font=fonts["hud_val"])
        draw.text((cx0 + cw - 32 - (pb[2] - pb[0]), cy0 + 16), f"{pct}%", font=fonts["hud_val"], fill=(251, 146, 60, int(255 * card_enter)))
        bx0, bx1, by0, by1 = cx0 + 32, cx0 + cw - 32, cy0 + 68, cy0 + 90
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=10, fill=(255, 255, 255, int(35 * card_enter)))
        draw.rounded_rectangle([bx0, by0, bx0 + int((bx1 - bx0) * fill_p), by1], radius=10, fill=(249, 115, 22, int(245 * card_enter)))

    elif hud_type == "timer_20":
        cw, ch = 430, 126
        cx0 = (W - cw) // 2
        cy0 = 180 + int((1.0 - card_enter) * 20)
        draw.rounded_rectangle(
            [cx0, cy0, cx0 + cw, cy0 + ch],
            radius=28,
            fill=(10, 18, 30, int(205 * card_enter)),
            outline=(20, 184, 166, int(220 * card_enter)),
            width=2,
        )
        rem = max(0, int(20 * 60 - int(local_t * 1.0)))
        mins, secs = divmod(rem, 60)
        t_str = f"{mins:02d}:{secs:02d}"
        tb = draw.textbbox((0, 0), t_str, font=fonts["timer"])
        draw.text((cx0 + (cw - (tb[2] - tb[0])) // 2, cy0 + 18), t_str, font=fonts["timer"], fill=(45, 212, 191, int(255 * card_enter)))

    elif hud_type == "closing_card":
        cw, ch = 860, 134
        cx0 = (W - cw) // 2
        cy0 = 180 + int((1.0 - card_enter) * 20)
        draw.rounded_rectangle(
            [cx0, cy0, cx0 + cw, cy0 + ch],
            radius=24,
            fill=(10, 15, 28, int(205 * card_enter)),
            outline=(245, 158, 11, int(200 * card_enter)),
            width=2,
        )
        steps = [("1. CAFÉ FORTE", (251, 191, 36)), ("2. 20 MIN EXATOS", (45, 212, 191)), ("3. ANTES DAS 15H", (96, 165, 250))]
        col_w = cw // 3
        for idx, (st, col) in enumerate(steps):
            sb = draw.textbbox((0, 0), st, font=fonts["hud_sm"])
            sx = cx0 + idx * col_w + (col_w - (sb[2] - sb[0])) // 2
            draw.text((sx, cy0 + 48), st, font=fonts["hud_sm"], fill=(col[0], col[1], col[2], int(255 * card_enter)))


def group_words_into_chunks(words: list[dict], max_words: int = 4, max_chars: int = 26) -> list[list[dict]]:
    """Groups word-level timestamps into balanced punchy 2-4 word Reels caption phrases."""
    chunks = []
    cur = []
    cur_chars = 0
    for w in words:
        wlen = len(w["word"])
        if cur and (len(cur) >= max_words or cur_chars + 1 + wlen > max_chars):
            chunks.append(cur)
            cur = [w]
            cur_chars = wlen
        else:
            cur.append(w)
            cur_chars += wlen + (1 if len(cur) > 1 else 0)
    if cur:
        # Avoid awkward 1-word orphan at the end if previous chunk has room
        if len(cur) == 1 and chunks and len(chunks[-1]) <= 3:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return chunks


def draw_kinetic_subtitles(draw: ImageDraw.ImageDraw, scene: dict, global_t: float, fonts: dict):
    """Draws bold, high-contrast kinetic Reels subtitles synced to the exact spoken word."""
    words = scene.get("words", [])
    if not words:
        return
    chunks = group_words_into_chunks(words, max_words=4, max_chars=25)
    highlights_lower = {h.lower().strip(".,;:!?") for h in scene.get("highlights", [])}

    # Find active chunk
    active_chunk = None
    for ci, ch in enumerate(chunks):
        c_start = ch[0]["start"] - 0.06
        c_end = ch[-1]["end"] + 0.16
        if ci + 1 < len(chunks):
            c_end = min(c_end, chunks[ci + 1][0]["start"] - 0.01)
        if c_start <= global_t <= c_end:
            active_chunk = ch
            break
    if active_chunk is None:
        return

    # Split chunk into 1 or 2 centered lines if > 19 chars
    lines = []
    cur_line = []
    cur_len = 0
    for w in active_chunk:
        if cur_line and cur_len + 1 + len(w["word"]) > 18:
            lines.append(cur_line)
            cur_line = [w]
            cur_len = len(w["word"])
        else:
            cur_line.append(w)
            cur_len += len(w["word"]) + (1 if len(cur_line) > 1 else 0)
    if cur_line:
        lines.append(cur_line)

    f_sub = fonts["sub"]
    f_sub_active = fonts["sub_active"]
    accent_rgb = hex_to_rgb(scene["badge_color"])

    line_height = 84
    base_y = 1545 if len(lines) == 2 else 1585

    for li, line in enumerate(lines):
        # Measure total line width
        rendered_tokens = []
        total_w = 0
        space_w = 18
        for wi, w in enumerate(line):
            text_up = w["word"].upper()
            is_spoken = (w["start"] - 0.04) <= global_t <= (w["end"] + 0.09)
            is_key = w["word"].lower().strip(".,;:!?—") in highlights_lower
            font_use = f_sub_active if is_spoken else f_sub
            bb = draw.textbbox((0, 0), text_up, font=font_use)
            tw = bb[2] - bb[0]
            th = bb[3] - bb[1]
            rendered_tokens.append((text_up, font_use, tw, th, is_spoken, is_key))
            total_w += tw + (space_w if wi < len(line) - 1 else 0)

        x = (W - total_w) // 2
        y = base_y + li * line_height

        for text_up, font_use, tw, th, is_spoken, is_key in rendered_tokens:
            y_off = -5 if is_spoken else 0
            if is_spoken:
                # Subtle pill highlight behind active spoken word
                pad = 12
                draw.rounded_rectangle(
                    [x - pad, y + y_off - 4, x + tw + pad, y + y_off + th + 16],
                    radius=16,
                    fill=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 70),
                )
                fill_col = (251, 191, 36, 255)  # Warm electric gold pop
            elif is_key:
                fill_col = (255, 255, 255, 255)
            else:
                fill_col = (241, 245, 249, 245)

            # Deep drop shadow + crisp stroke for 100% legibility on mobile
            draw.text(
                (x, y + y_off + 5),
                text_up,
                font=font_use,
                fill=(0, 0, 0, 190),
                stroke_width=7,
                stroke_fill=(0, 0, 0, 210),
            )
            draw.text(
                (x, y + y_off),
                text_up,
                font=font_use,
                fill=fill_col,
                stroke_width=6,
                stroke_fill=(8, 12, 22, 255),
            )
            x += tw + space_w


def render_single_scene(args: tuple) -> str:
    """Worker function: renders one 5.0-second scene (150 frames) into a crisp lossless segment MP4."""
    scene_idx, timeline = args
    scenes = timeline["scenes"]
    scene = scenes[scene_idx]
    total_scenes = len(scenes)
    accent_rgb = hex_to_rgb(scene["badge_color"])

    fonts = {
        "badge": ImageFont.truetype(FONT_BOLD, 30),
        "hud_sm": ImageFont.truetype(FONT_BOLD, 25),
        "hud_xs": ImageFont.truetype(FONT_SEMIBOLD, 21),
        "hud_val": ImageFont.truetype(FONT_BLACK, 34),
        "timer": ImageFont.truetype(FONT_BLACK, 72),
        "sub": ImageFont.truetype(FONT_BLACK, 58),
        "sub_active": ImageFont.truetype(FONT_BLACK, 64),
    }

    cur_base = load_prescaled_image(scene["image"])
    prev_base = load_prescaled_image(scenes[scene_idx - 1]["image"]) if scene_idx > 0 else None
    prev_motion = scenes[scene_idx - 1]["motion"] if scene_idx > 0 else None

    scrim_mult, _ = build_scrim_lut()
    particles = build_particles(scene["id"], count=22)

    seg_path = SEGMENTS_DIR / f"seg_{scene_idx + 1:02d}.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        str(seg_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    for f_idx in range(FRAMES_PER_SCENE):
        local_t = f_idx / float(FPS)
        global_t = scene["start"] + local_t
        progress = f_idx / float(FRAMES_PER_SCENE - 1)

        # Sample current scene frame with smooth gimbal camera motion
        frame_img = sample_camera_frame(cur_base, progress, scene["motion"])
        frame_arr = np.asarray(frame_img, dtype=np.float32)

        # Smooth 0.20s cross-dissolve + subtle geometric light flash from previous scene
        if prev_base is not None and f_idx < TRANSITION_FRAMES:
            alpha = ease_in_out((f_idx + 1) / float(TRANSITION_FRAMES + 1))
            prev_frame = sample_camera_frame(prev_base, 1.0, prev_motion)
            prev_arr = np.asarray(prev_frame, dtype=np.float32)
            frame_arr = prev_arr * (1.0 - alpha) + frame_arr * alpha
            # Crisp subtle prism flash peaking at midpoint of transition
            flash = math.sin(alpha * math.pi) * 26.0
            frame_arr = np.clip(frame_arr + flash, 0.0, 255.0)
        elif scene_idx == 0 and f_idx < 8:
            # Smooth crisp fade-in on very first frames
            fade_in = (f_idx + 1) / 8.0
            frame_arr *= fade_in
        elif scene_idx == total_scenes - 1 and f_idx >= FRAMES_PER_SCENE - 12:
            # Gentle cinematic fade-out on final 0.4s
            rem = (FRAMES_PER_SCENE - 1 - f_idx) / 12.0
            frame_arr *= max(0.0, rem)

        # Apply cinematic top & bottom legibility scrim
        frame_arr *= scrim_mult

        # Composite RGBA overlay (floating 3D geometric dust + progress bar + HUD + kinetic subtitles)
        base_rgba = Image.fromarray(frame_arr.astype(np.uint8), mode="RGB").convert("RGBA")
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_particles(overlay, particles, local_t, accent_rgb)
        ov_draw = ImageDraw.Draw(overlay, "RGBA")
        draw_progress_bar(ov_draw, scene_idx, total_scenes, progress, accent_rgb)
        draw_hud(ov_draw, scene, local_t, fonts)
        draw_kinetic_subtitles(ov_draw, scene, global_t, fonts)

        composed = Image.alpha_composite(base_rgba, overlay).convert("RGB")
        proc.stdin.write(composed.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed on scene {scene_idx + 1}")
    print(f"Rendered segment {scene_idx + 1:02d}/{total_scenes} -> {seg_path.name}")
    return str(seg_path)


def main():
    t0 = time.time()
    with open(TIMELINE_PATH, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    total_scenes = len(timeline["scenes"])
    print(f"Starting parallel render of {total_scenes} scenes ({total_scenes * SCENE_DUR:.1f}s @ {FPS}fps, {W}x{H})...")

    tasks = [(i, timeline) for i in range(total_scenes)]
    workers = min(os.cpu_count() or 2, 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        seg_paths = list(ex.map(render_single_scene, tasks))

    # Concatenate all 16 segments losslessly and mux mastered stereo audio
    concat_txt = BASE_DIR / "build" / "concat.txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for sp in seg_paths:
            f.write(f"file '{sp}'\n")

    final_mp4 = OUTPUT_DIR / "coffee_nap_3d_poligonos_80s.mp4"
    print("Concatenating segments & muxing mastered audio...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "warning",
            "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-i", str(AUDIO_PATH),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            "-shortest",
            str(final_mp4),
        ],
        check=True,
    )

    elapsed = time.time() - t0
    size_mb = final_mp4.stat().st_size / (1024 * 1024)
    print(f"\nDONE! Rendered {final_mp4} ({size_mb:.2f} MB) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
