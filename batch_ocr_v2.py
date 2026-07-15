"""
批量 OCR，保留位置信息 → 输出结构化 Markdown
"""
import os
import time
import numpy as np

# === 模型存 E 盘 ===
_CACHE = r"E:\soft\OCR\Paddle_ocr\models"
os.environ["PADDLE_PDX_CACHE_HOME"] = _CACHE
os.environ["PADDLEX_HOME"] = _CACHE

from paddleocr import PaddleOCR

PROJECT_DIR = r"E:\soft\OCR\Paddle_ocr"

# 找到所有截图
images = sorted([
    f for f in os.listdir(PROJECT_DIR)
    if f.startswith("Snipaste") and f.endswith(".png")
])

print(f"找到 {len(images)} 张图片\n加载模型...")
ocr = PaddleOCR(lang="ch")
print("开始识别\n")

md_lines = [
    "# PaddleOCR 批量识别结果（含位置信息）",
    "",
    f"**识别时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"**模型**: PP-OCRv6 medium  |  **设备**: GPU (RTX 4070 Ti SUPER)",
    "",
    "> 文字按从上到下、从左到右排列，坐标格式：`(x₁,y₁)-(x₂,y₂)` 为包围框左上→右下角",
    "",
    "---",
    "",
]

# 同一行的 Y 容差（像素）
LINE_Y_TOLERANCE = 15

for i, img_name in enumerate(images, 1):
    img_path = os.path.join(PROJECT_DIR, img_name)
    print(f"[{i}/{len(images)}] {img_name}")

    t0 = time.time()
    result = ocr.predict(img_path)
    elapsed = time.time() - t0

    page = result[0]
    texts = page["rec_texts"]
    scores = page["rec_scores"]
    polys = page["dt_polys"]  # shape (4, 2)

    # 计算每个文本框的中心 Y 和左边界 X
    items = []
    for j, (text, score, poly) in enumerate(zip(texts, scores, polys)):
        y_center = float(np.mean(poly[:, 1]))
        x_left = float(np.min(poly[:, 0]))
        x_right = float(np.max(poly[:, 0]))
        y_top = float(np.min(poly[:, 1]))
        y_bottom = float(np.max(poly[:, 1]))
        items.append({
            "text": text,
            "score": score,
            "y_center": y_center,
            "x_left": x_left,
            "x_right": x_right,
            "y_top": y_top,
            "y_bottom": y_bottom,
        })

    # 按 Y 中心排序（上→下），同 Y 按 X 排序（左→右）
    items.sort(key=lambda it: (round(it["y_center"] / LINE_Y_TOLERANCE), it["x_left"]))

    # 图片尺寸
    img_h = int(max(it["y_bottom"] for it in items)) if items else 0
    img_w = int(max(it["x_right"] for it in items)) if items else 0

    md_lines.append(f"## {i}. {img_name}")
    md_lines.append("")
    md_lines.append(f"![{img_name}]({img_name})")
    md_lines.append("")
    md_lines.append(f"**耗时**: {elapsed:.1f}s  |  **图片尺寸**: {img_w}×{img_h}  |  **识别条数**: {len(items)}")
    md_lines.append("")

    if items:
        md_lines.append("| # | 文字 | 位置 (左上→右下) | 置信度 |")
        md_lines.append("|---|------|-------------------|--------|")
        for j, it in enumerate(items, 1):
            pos = f"({it['x_left']:.0f},{it['y_top']:.0f})-({it['x_right']:.0f},{it['y_bottom']:.0f})"
            md_lines.append(f"| {j} | {it['text']} | {pos} | {it['score']:.1%} |")
        md_lines.append("")

        # 纯文本段落（方便 AI 阅读）
        md_lines.append("<details>")
        md_lines.append("<summary>纯文本（按阅读顺序）</summary>")
        md_lines.append("")
        md_lines.append("```")
        current_y_bucket = round(items[0]["y_center"] / LINE_Y_TOLERANCE)
        line_texts = []
        for it in items:
            y_bucket = round(it["y_center"] / LINE_Y_TOLERANCE)
            if y_bucket != current_y_bucket:
                md_lines.append("  ".join(line_texts))
                line_texts = []
                current_y_bucket = y_bucket
            line_texts.append(it["text"])
        if line_texts:
            md_lines.append("  ".join(line_texts))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("</details>")
        md_lines.append("")
    else:
        md_lines.append("> 未识别到文字")
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("")

# 保存
output_path = os.path.join(PROJECT_DIR, "ocr_results.md")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\n结果已保存: {output_path}")
