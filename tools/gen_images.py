"""히어로 배경과 카테고리별 카드 커버를 생성한다 (한 번 돌리고 결과 PNG 를 커밋).

usage: python tools/gen_images.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "img"
MONO = "/System/Library/Fonts/SFNSMono.ttf"
BG = (11, 15, 20)

# 카테고리 → (파일명, 주 색, 보조 색, 라벨)
COVERS = {
    "AI Engineering": ("ai", (52, 211, 153), (34, 211, 238), "AI ENGINEERING"),
    "ML & Modeling": ("ml", (129, 140, 248), (192, 132, 252), "ML & MODELING"),
    "MLOps & Infra": ("ops", (56, 189, 248), (45, 212, 191), "MLOPS & INFRA"),
    "Dev Notes": ("dev", (251, 191, 36), (251, 146, 60), "DEV NOTES"),
    "Troubleshooting": ("trouble", (248, 113, 113), (251, 146, 60), "TROUBLESHOOTING"),
    "Paper Review": ("paper", (244, 114, 182), (167, 139, 250), "PAPER REVIEW"),
    "Retrospective": ("retro", (163, 230, 53), (52, 211, 153), "RETROSPECTIVE"),
}


def glow_blob(size, center, radius, color, strength):
    """가우시안 광원 한 개를 RGB 레이어로 만든다."""
    w, h = size
    y, x = np.ogrid[:h, :w]
    d2 = (x - center[0]) ** 2 + (y - center[1]) ** 2
    a = np.exp(-d2 / (2 * radius**2)) * strength
    return np.stack([a * c for c in color], axis=-1)


def network(draw, rng, w, h, n, color, max_d, node_r=(1.2, 3.2), edge_alpha=70):
    pts = np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h, n)])
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(*(pts[i] - pts[j]))
            if d < max_d:
                a = int(edge_alpha * (1 - d / max_d))
                draw.line([tuple(pts[i]), tuple(pts[j])], fill=color + (a,), width=1)
    for p in pts:
        r = rng.uniform(*node_r)
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color + (220,))


def base(size, c1, c2, seed):
    w, h = size
    rng = np.random.default_rng(seed)
    arr = np.zeros((h, w, 3)) + np.array(BG, dtype=float)
    arr += glow_blob(size, (w * 0.80, h * 0.20), w * 0.22, c1, 0.42)
    arr += glow_blob(size, (w * 0.10, h * 1.00), w * 0.24, c2, 0.30)
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGBA")
    return img, rng


def grid(draw, w, h, step, alpha):
    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, alpha))
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, alpha))


def hero():
    w, h = 2400, 1400
    img, rng = base((w, h), (52, 211, 153), (34, 211, 238), 7)
    layer = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(layer)
    grid(d, w, h, 80, 10)
    network(d, rng, w, h, 170, (110, 231, 183), 200, node_r=(1.5, 3.6), edge_alpha=90)
    glow = layer.filter(ImageFilter.GaussianBlur(6))
    img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, layer)
    img.convert("RGB").save(OUT / "hero.jpg", quality=88)


def cover(name, c1, c2, label, seed):
    w, h = 1200, 800
    img, rng = base((w, h), c1, c2, seed)
    layer = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(layer)
    grid(d, w, h, 48, 12)
    network(d, rng, w, h, 46, c1, 170, edge_alpha=80)
    img = Image.alpha_composite(img, layer.filter(ImageFilter.GaussianBlur(3)))
    img = Image.alpha_composite(img, layer)
    t = ImageDraw.Draw(img)
    font = ImageFont.truetype(MONO, 60)
    small = ImageFont.truetype(MONO, 26)
    t.text((64, h - 176), "// " + label.lower().replace(" ", "_"), font=small, fill=c1 + (255,))
    t.text((64, h - 132), label, font=font, fill=(236, 240, 244, 255))
    img.convert("RGB").save(OUT / "covers" / f"{name}.jpg", quality=86)


if __name__ == "__main__":
    (OUT / "covers").mkdir(parents=True, exist_ok=True)
    hero()
    for i, (cat, (name, c1, c2, label)) in enumerate(COVERS.items()):
        cover(name, c1, c2, label, 100 + i)
    # Veil 리포트 전용 커버
    cover("veil", (52, 211, 153), (129, 140, 248), "VEIL-PII-KO-LITE", 42)
    print("ok")
