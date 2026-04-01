#!/usr/bin/env python3
"""Generate side-by-side comparison video: GT | Shifted | Reverse-shifted.

Usage:
    python generate_comparison_video.py \
        --shifted-root  /path/to/shifted/sensor/train/<log_id> \
        --reverse-root  /path/to/reverse_shifted/sensor/train/<log_id> \
        --output-dir   /path/to/scene_root \
        --height 480
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int = 22) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def generate(
    shifted_root: str,
    reverse_root: str,
    output_dir: str,
    target_h: int = 480,
    fps: int = 10,
) -> list[str]:
    """Return list of written video paths."""
    font = _load_font()
    # Same layout as pairs Step 6: list frames from reverse sensors; GT from shifted_root/gt/<cam>/
    cam_base = os.path.join(reverse_root, "sensors", "cameras")
    if not os.path.isdir(cam_base):
        print(f"[warn] cameras dir not found: {cam_base}", file=sys.stderr)
        return []

    cameras = sorted(
        d for d in os.listdir(cam_base)
        if os.path.isdir(os.path.join(cam_base, d))
    )

    written: list[str] = []
    for cam in cameras:
        gt_dir = os.path.join(shifted_root, "gt", cam)
        sh_dir = os.path.join(shifted_root, "sensors", "cameras", cam)
        rv_dir = os.path.join(reverse_root, "sensors", "cameras", cam)

        if not os.path.isdir(rv_dir):
            continue
        timestamps = sorted(f for f in os.listdir(rv_dir) if f.endswith(".jpg"))
        if not timestamps:
            continue

        frames: list[np.ndarray] = []
        for ts in timestamps:
            paths = (
                os.path.join(gt_dir, ts),
                os.path.join(sh_dir, ts),
                os.path.join(rv_dir, ts),
            )
            if not all(os.path.exists(p) for p in paths):
                continue

            imgs = [Image.open(p).convert("RGB") for p in paths]  # gt, shifted, reverse
            w0, h0 = imgs[0].size
            ratio = target_h / h0
            tw = int(w0 * ratio) & ~1
            th = target_h & ~1

            labels = ["GT", "Shifted", "Reverse-shifted"]
            resized = []
            for img, label in zip(imgs, labels):
                img = img.resize((tw, th), Image.LANCZOS)
                draw = ImageDraw.Draw(img)
                bbox = draw.textbbox((0, 0), label, font=font)
                lw = bbox[2] - bbox[0] + 16
                lh = bbox[3] - bbox[1] + 12
                draw.rectangle([0, 0, lw, lh], fill=(0, 0, 0, 180))
                draw.text((8, 4), label, fill="white", font=font)
                resized.append(img)

            combined = Image.new("RGB", (tw * 3, th))
            for i, img in enumerate(resized):
                combined.paste(img, (tw * i, 0))
            frames.append(np.array(combined))

        if not frames:
            print(f"  [skip] {cam}: no matching triplets")
            continue

        out_path = os.path.join(output_dir, f"comparison_{cam}.mp4")
        try:
            import mediapy as media
            media.write_video(out_path, frames, fps=fps)
        except ImportError:
            import imageio
            writer = imageio.get_writer(
                out_path, fps=fps, codec="libx264",
                output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
            )
            for f in frames:
                writer.append_data(f)
            writer.close()

        print(f"  [done] {cam}: {len(frames)} frames -> {out_path}")
        written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GT|Shifted|Reverse comparison video")
    parser.add_argument(
        "--shifted-root",
        required=True,
        help="Shifted scene root (uses gt/<cam>/ and sensors/cameras/<cam>/ from render_shifted_av2)",
    )
    parser.add_argument("--reverse-root", required=True, help="Path to reverse-shifted scene")
    parser.add_argument("--output-dir", required=True, help="Directory for output .mp4 files")
    parser.add_argument("--height", type=int, default=480, help="Per-panel height in pixels")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    written = generate(
        shifted_root=args.shifted_root,
        reverse_root=args.reverse_root,
        output_dir=args.output_dir,
        target_h=args.height,
        fps=args.fps,
    )
    print(f"Videos saved: {len(written)} files in {args.output_dir}")


if __name__ == "__main__":
    main()
