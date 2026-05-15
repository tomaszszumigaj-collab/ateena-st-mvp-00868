from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from body_analysis import analyze_body


def make_silhouette(front: bool = True) -> bytes:
    h, w = 1200, 700
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    pts = np.array([
        [350, 110], [315, 155], [300, 250], [260, 330], [270, 530], [300, 680], [290, 1040], [320, 1160],
        [380, 1160], [410, 1040], [400, 680], [430, 530], [440, 330], [400, 250], [385, 155]
    ], np.int32) if front else np.array([
        [340, 110], [320, 150], [315, 250], [300, 330], [305, 520], [325, 680], [315, 1040], [330, 1160],
        [370, 1160], [380, 1040], [375, 680], [400, 520], [415, 350], [405, 250], [390, 150]
    ], np.int32)
    cv2.fillPoly(img, [pts], (30, 30, 30))
    cv2.circle(img, (350, 85), 52 if front else 46, (30, 30, 30), -1)
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise RuntimeError('Encoding failed')
    return buf.tobytes()


def main():
    front = make_silhouette(front=True)
    profile = make_silhouette(front=False)
    result = analyze_body(front, profile, 170, 62, 30)
    print(result.to_dict())


if __name__ == '__main__':
    main()
