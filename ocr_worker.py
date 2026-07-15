"""
PaddleOCR 子进程 Worker
每次 OCR 调用在独立进程中执行，完成后自动释放所有 GPU 资源。
用法: python ocr_worker.py <image_path>
输出: JSON（单行）到 stdout
"""
import os
import sys
import json

# ── 环境配置 ──────────────────────────────────────────────────
_CACHE = r"E:\soft\OCR\Paddle_ocr\models"
os.environ["PADDLE_PDX_CACHE_HOME"] = _CACHE
os.environ["PADDLEX_HOME"] = _CACHE
os.environ["MODELSCOPE_CACHE"] = _CACHE
os.makedirs(_CACHE, exist_ok=True)

_ENV = r"E:\soft\anaconda3\envs\paddle_ocr\Lib\site-packages"
_dll_dirs = [
    r"nvidia\cudnn\bin", r"nvidia\cuda_runtime\bin", r"nvidia\cublas\bin",
    r"nvidia\cufft\bin", r"nvidia\curand\bin", r"nvidia\cusparse\bin",
    r"nvidia\cusolver\bin", r"torch\lib",
]
for d in _dll_dirs:
    full = os.path.join(_ENV, d)
    if os.path.isdir(full):
        os.add_dll_directory(full)

# ── OCR ───────────────────────────────────────────────────────
from paddleocr import PaddleOCR

def main():
    if len(sys.argv) < 2:
        json.dump({"error": "usage: ocr_worker.py <image_path>"}, sys.stdout)
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        json.dump({"error": f"file not found: {image_path}"}, sys.stdout)
        sys.exit(1)

    ocr = PaddleOCR(lang="ch", use_textline_orientation=False)
    result = ocr.predict(image_path)

    texts = []
    for page in result:
        rec_texts = page.get("rec_texts", [])
        rec_scores = page.get("rec_scores", [])
        for i, item in enumerate(rec_texts):
            if isinstance(item, dict):
                texts.append({
                    "text": item.get("text", ""),
                    "confidence": round(item.get("score", 0), 4),
                })
            elif isinstance(item, str):
                score = rec_scores[i] if i < len(rec_scores) else 1.0
                texts.append({"text": item, "confidence": round(score, 4)})

    output = {
        "image": os.path.basename(image_path),
        "text_count": len(texts),
        "texts": texts,
        "full_text": "\n".join(t["text"] for t in texts),
    }
    json.dump(output, sys.stdout)

if __name__ == "__main__":
    main()
