"""
输出 PaddleOCR predict() 返回的全部字段，含说明
"""
import os, json, time
import numpy as np

os.environ["PADDLE_PDX_CACHE_HOME"] = r"E:\soft\OCR\Paddle_ocr\models"
from paddleocr import PaddleOCR

PROJECT_DIR = r"E:\soft\OCR\Paddle_ocr"
images = sorted(f for f in os.listdir(PROJECT_DIR) if f.startswith("Snipaste") and f.endswith(".png"))

print(f"加载模型，识别 {len(images)} 张图...")
ocr = PaddleOCR(lang="ch")

md = [
    "# PaddleOCR `predict()` 全部返回字段详解",
    "",
    f"**模型**: PP-OCRv6 medium  |  **设备**: RTX 4070 Ti SUPER (GPU)  |  **时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "---",
    "",
    "## 字段总览",
    "",
    "`predict()` 返回 `list[dict]`，每页一个 dict，含以下 15 个字段：",
    "",
    "### 核心识别",
    "| 字段 | 类型 | 说明 |",
    "|------|------|------|",
    "| `rec_texts` | `list[str]` | 识别出的文字内容 |",
    "| `rec_scores` | `list[float]` | 置信度 (0~1)，越高越可信 |",
    "| `dt_polys` | `list[ndarray(4,2)]` | 文本检测框，4 个角点坐标 `[[x0,y0],[x1,y1],[x2,y2],[x3,y3]]` |",
    "| `rec_polys` | `list[ndarray(4,2)]` | 文本识别框（矫正后），比检测框更精确 |",
    "| `rec_boxes` | `ndarray(N,4)` | 紧凑包围盒 `(x, y, w, h)` — 左上角+宽高 |",
    "",
    "### 文档预处理",
    "| 字段 | 类型 | 说明 |",
    "|------|------|------|",
    "| `doc_preprocessor_res` | `dict` | 文档预处理结果：含原始图像、矫正图像、旋转角度、是否触发矫正等 |",
    "| `textline_orientation_angles` | `list[int]` | 每行文字的方向角：`0`=正向 `90`=右旋 `180`=倒置 `270`=左旋 |",
    "| `model_settings` | `dict` | 记录预处理开关状态 |",
    "",
    "### 检测参数",
    "| 字段 | 类型 | 说明 |",
    "|------|------|------|",
    "| `text_det_params` | `dict` | 文本检测参数：边长限制、阈值、框阈值等 |",
    "| `text_type` | `str` | 文本类型，固定为 `\"general\"` |",
    "| `text_rec_score_thresh` | `float` | 识别分数阈值，低于此值的文字会被丢弃 |",
    "| `return_word_box` | `bool` | 是否返回单词级别包围框（默认 False） |",
    "",
    "### 元信息",
    "| 字段 | 类型 | 说明 |",
    "|------|------|------|",
    "| `input_path` | `str` | 输入图片的完整路径 |",
    "| `page_index` | `None\\|int` | PDF 的页码，图片为 None |",
    "| `vis_fonts` | `list[Font]` | 可视化用的字体对象列表（用于渲染识别结果） |",
    "",
    "---",
    "",
]

for idx, img_name in enumerate(images, 1):
    img_path = os.path.join(PROJECT_DIR, img_name)
    print(f"[{idx}/{len(images)}] {img_name}")
    t0 = time.time()
    result = ocr.predict(img_path)
    page = result[0]
    elapsed = time.time() - t0

    md.append(f"## 图{idx}. {img_name}")
    md.append("")
    md.append(f"![{img_name}]({img_name})")
    md.append(f"*耗时 {elapsed:.1f}s，识别 {len(page['rec_texts'])} 条文字*")
    md.append("")

    # === 1. input_path ===
    md.append("### `input_path` — 输入路径")
    md.append("")
    md.append(f"```\n{page['input_path']}\n```")
    md.append("")

    # === 2. page_index ===
    md.append("### `page_index` — 页码")
    md.append("")
    md.append(f"```\n{page['page_index']}\n```")
    md.append("> 图片为 `None`，PDF 多页时为页码序号")
    md.append("")

    # === 3. doc_preprocessor_res ===
    md.append("### `doc_preprocessor_res` — 文档预处理")
    md.append("")
    prep = page["doc_preprocessor_res"]
    if prep and hasattr(prep, '__dict__'):
        for k, v in prep.__dict__.items():
            if isinstance(v, np.ndarray):
                md.append(f"- **{k}**: ndarray shape={v.shape} dtype={v.dtype}")
            else:
                md.append(f"- **{k}**: {v}")
    elif prep and isinstance(prep, dict):
        for k, v in prep.items():
            if isinstance(v, np.ndarray):
                md.append(f"- **{k}**: ndarray shape={v.shape} dtype={v.dtype}")
            else:
                md.append(f"- **{k}**: {v}")
    md.append("")

    # === 4. dt_polys ===
    md.append("### `dt_polys` — 检测框坐标")
    md.append("")
    md.append(f"共 `{len(page['dt_polys'])}` 个框，每个是 `(4,2)` 的 numpy 数组（4个角点 x,y）：")
    md.append("")
    for j, poly in enumerate(page["dt_polys"][:3]):
        md.append(f"- 框[{j}]: `{poly.tolist()}` → 文字: `{page['rec_texts'][j][:60]}`")
    if len(page["dt_polys"]) > 3:
        md.append(f"- ... 共 {len(page['dt_polys'])} 个")
    md.append("")

    # === 5. model_settings ===
    md.append("### `model_settings` — 模型设置")
    md.append("")
    md.append("```json")
    md.append(json.dumps(page["model_settings"], indent=2, ensure_ascii=False, default=str))
    md.append("```")
    md.append("")

    # === 6. text_det_params ===
    md.append("### `text_det_params` — 文本检测参数")
    md.append("")
    md.append("```json")
    md.append(json.dumps(page["text_det_params"], indent=2, ensure_ascii=False, default=str))
    md.append("```")
    md.append("")

    # === 7. text_type ===
    md.append("### `text_type` — 文本类型")
    md.append("")
    md.append(f"```\n{page['text_type']}\n```")
    md.append("")

    # === 8. text_rec_score_thresh ===
    md.append("### `text_rec_score_thresh` — 识别分数阈值")
    md.append("")
    md.append(f"```\n{page['text_rec_score_thresh']}\n```")
    md.append("> 低于此值的识别结果被丢弃。`0.0` 表示不过滤")
    md.append("")

    # === 9. return_word_box ===
    md.append("### `return_word_box` — 是否返回词级包围框")
    md.append("")
    md.append(f"```\n{page['return_word_box']}\n```")
    md.append("")

    # === 10-12. rec_texts / rec_scores / rec_polys ===
    items_count = len(page["rec_texts"])
    md.append("### `rec_texts` / `rec_scores` / `rec_polys` — 识别三件套")
    md.append("")
    md.append(f"共 `{items_count}` 条。前 10 条：")
    md.append("")
    md.append("| # | rec_texts | rec_scores | rec_polys (左上角) |")
    md.append("|---|-----------|------------|---------------------|")
    for j in range(min(10, items_count)):
        text = page["rec_texts"][j][:50]
        score = page["rec_scores"][j]
        rp = page["rec_polys"][j]
        corner = f"({rp[0,0]:.0f},{rp[0,1]:.0f})"
        md.append(f"| {j} | {text} | {score:.1%} | {corner} |")
    if items_count > 10:
        md.append(f"| ... | ... | ... | ... |")
    md.append("")

    # === 13. vis_fonts ===
    md.append("### `vis_fonts` — 可视化字体")
    md.append("")
    md.append(f"共 `{len(page['vis_fonts'])}` 个 Font 对象，用于绘制识别结果")
    md.append("")

    # === 14. textline_orientation_angles ===
    md.append("### `textline_orientation_angles` — 文本行方向角")
    md.append("")
    angles = page["textline_orientation_angles"]
    unique = set(angles)
    md.append(f"全 `{len(angles)}` 行，取值分布: {dict((a, angles.count(a)) for a in sorted(unique))}")
    md.append("")
    md.append("> `0`=正向 `90`=右旋 `180`=倒置 `270`=左旋")
    md.append("")

    # === 15. rec_boxes ===
    md.append("### `rec_boxes` — 包围盒 (x,y,w,h)")
    md.append("")
    md.append(f"ndarray shape=`{page['rec_boxes'].shape}` dtype=`{page['rec_boxes'].dtype}`")
    md.append("")
    md.append("前 5 行 `[x, y, w, h]`：")
    md.append("")
    md.append("| # | x | y | w | h | 对应文字 |")
    md.append("|---|---|---|---|---|----------|")
    for j in range(min(5, items_count)):
        x, y, w, h = page["rec_boxes"][j]
        text = page["rec_texts"][j][:30]
        md.append(f"| {j} | {x} | {y} | {w} | {h} | {text} |")
    md.append("")

    md.append("---")
    md.append("")

# 保存
output_path = os.path.join(PROJECT_DIR, "ocr_full_fields.md")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"\n已保存: {output_path}")
