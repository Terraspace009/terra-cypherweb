import cv2
import numpy as np
from PIL import Image, ImageDraw

# Cyberpunk palette (from image)
ascii_colors = [
    (92, 88, 182),   # #5c58b6 (purple-blue)
    (185, 87, 206),  # #b957ce (pinkish purple)
    (89, 148, 206),  # #5994ce (light blue)
    (58, 78, 147),   # #3a4e93 (dark blue)
    (255, 255, 255)  # white highlight
]

_prev_ascii_frame = None

def frame_to_matrix_dots(image, cols=90, dot_size=7, smoothing_alpha=0.7, prev_gray=None, emotion="neutral"):
    global _prev_ascii_frame

    h, w = image.shape[:2]
    cell_w = w / cols
    rows = int(h / cell_w)
    img_resized = cv2.resize(image, (cols, rows))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    if _prev_ascii_frame is None:
        _prev_ascii_frame = gray.copy()
    elif _prev_ascii_frame.shape != gray.shape:
        _prev_ascii_frame = cv2.resize(_prev_ascii_frame, (gray.shape[1], gray.shape[0]))

    blended = cv2.addWeighted(gray, 0.9, _prev_ascii_frame, 0.1, 0)

    _prev_ascii_frame = blended.copy()

    output = Image.new("RGB", (cols * dot_size, rows * dot_size), (0, 0, 0))
    draw = ImageDraw.Draw(output)

    for y in range(rows):
        for x in range(cols):
            brightness = blended[y, x]
            radius = int((brightness / 255) * (dot_size / 2))
            if radius > 0:
                color_index = int((brightness / 255) * (len(ascii_colors) - 1))
                color = ascii_colors[color_index]
                cx = x * dot_size + dot_size // 2
                cy = y * dot_size + dot_size // 2
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)

    return output, gray

