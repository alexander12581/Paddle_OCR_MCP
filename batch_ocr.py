"""
批量 OCR 识别当前目录下所有 PNG 图片，输出到 results.md
"""
import os
import sys
import time

# === 模型存 E 盘 ===
_CACHE = r"E:\soft\OCR\Paddle_ocr\models"
os.environ["PADDLE_PDX_CACHE_HOME"] = _CACHE
os.environ["PADDLEX_HOME"] = _CACHE

from paddleocr import PaddleOCR

PROJECT_DIR = r"E:\soft\OCR\Paddle_ocr"

# 找到所有 PNG（排除 test_images 目录和我们生成的文件）
images = sorted([
    f for f in os.listdir(PROJECT_DIR)
    if f.endswith('.png') and not f.startswith('test_')
])

# 排除 test_images 里的
images = [f for f in images if os.path.isfile(os.path.join(PROJECT_DIR, f))]

print(f"找到 {len(images)} 张图片:")
for img in images:
    print(f"  - {img}")

# 初始化 OCR
print("\n加载 PP-OCRv6 模型...")
ocr = PaddleOCR(lang='ch')
print("模型就绪\n")

# 批量识别
md_lines = [
    "# PaddleOCR 批量识别结果",
    "",
    f"识别时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"模型: PP-OCRv6 medium",
    f"设备: GPU (RTX 4070 Ti SUPER)",
    f"图片数量: {len(images)}",
    "",
    "---",
    "",
]

for i, img_name in enumerate(images, 1):
    img_path = os.path.join(PROJECT_DIR, img_name)
    print(f"[{i}/{len(images)}] 识别: {img_name}")

    t0 = time.time()
    result = ocr.predict(img_path)
    elapsed = time.time() - t0

    md_lines.append(f"## {i}. {img_name}")
    md_lines.append("")
    md_lines.append(f"![{img_name}]({img_name})")
    md_lines.append("")
    md_lines.append(f"*耗时: {elapsed:.1f}s*")
    md_lines.append("")

    texts_found = []
    for page in result:
        rec_texts = page.get("rec_texts", [])
        for item in rec_texts:
            if isinstance(item, dict):
                text = item.get("text", "")
                score = item.get("score", 0)
                texts_found.append((text, score))
            elif isinstance(item, str):
                texts_found.append((item, 1.0))

    if texts_found:
        md_lines.append("| 序号 | 文字 | 置信度 |")
        md_lines.append("|------|------|--------|")
        for j, (text, score) in enumerate(texts_found, 1):
            md_lines.append(f"| {j} | {text} | {score:.2%} |")
        md_lines.append("")
    else:
        md_lines.append("> 未识别到文字")
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("")

# 保存
output_path = os.path.join(PROJECT_DIR, "ocr_results.md")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f"\n结果已保存至: {output_path}")
