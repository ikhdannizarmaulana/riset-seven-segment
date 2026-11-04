"""Buat citra uji sintetis angka seven segment dengan PIL (bukan dari font).

Contoh: .venv/bin/python tools/make_test_image.py 123.4 data/test_1234.png --style lcd
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

#      a
#    f   b
#      g
#    e   c
#      d
SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}


def _seg_rects(x: int, y: int, w: int, h: int, t: int, gap: int) -> dict[str, tuple]:
    hh = h // 2
    return {
        "a": (x + t + gap, y, x + w - t - gap, y + t),
        "g": (x + t + gap, y + hh - t // 2, x + w - t - gap, y + hh + t // 2),
        "d": (x + t + gap, y + h - t, x + w - t - gap, y + h),
        "f": (x, y + t + gap, x + t, y + hh - gap),
        "b": (x + w - t, y + t + gap, x + w, y + hh - gap),
        "e": (x, y + hh + gap, x + t, y + h - t - gap),
        "c": (x + w - t, y + hh + gap, x + w, y + h - t - gap),
    }


def render(text: str, digit_w: int = 60, digit_h: int = 110, thick: int = 12,
           style: str = "lcd", margin: int = 40) -> Image.Image:
    if style == "lcd":
        bg, on, off = (170, 190, 160), (25, 30, 25), (155, 175, 147)
    else:  # led: merah menyala di latar gelap
        bg, on, off = (20, 10, 10), (255, 40, 30), (45, 15, 15)
    spacing, dot_w = 22, 18
    n_digits = sum(ch.isdigit() for ch in text)
    width = margin * 2 + n_digits * (digit_w + spacing) + text.count(".") * dot_w
    img = Image.new("RGB", (width, digit_h + margin * 2), bg)
    d = ImageDraw.Draw(img)
    x = margin
    for ch in text:
        if ch == ".":
            d.rectangle((x - spacing // 2, margin + digit_h - thick,
                         x - spacing // 2 + thick, margin + digit_h), fill=on)
            x += dot_w
            continue
        rects = _seg_rects(x, margin, digit_w, digit_h, thick, gap=2)
        for name, r in rects.items():
            d.rectangle(r, fill=on if name in SEGMENTS[ch] else off)
        x += digit_w + spacing
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--style", choices=["lcd", "led"], default="lcd")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    render(a.text, style=a.style).save(a.out)
    print(f"tersimpan: {a.out}")
