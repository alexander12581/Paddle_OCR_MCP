"""
PaddleOCR PP-OCRv6 — 最终可用版本
使用方法: python run_ocr.py <图片路径>
"""
import os
import sys

# === 模型存 E 盘 ===
_CACHE = r"E:\soft\OCR\Paddle_ocr\models"
os.environ["PADDLE_PDX_CACHE_HOME"] = _CACHE
os.environ["PADDLEX_HOME"] = _CACHE
os.environ["MODELSCOPE_CACHE"] = _CACHE
os.makedirs(_CACHE, exist_ok=True)

from paddleocr import PaddleOCR

# 初始化（首次会自动下载模型）
ocr = PaddleOCR(lang='ch')

def ocr_image(image_path):
    """识别图片，返回文字列表"""
    # 新版 API 用 predict
    result = ocr.predict(image_path)

    texts = []
    for page in result:
        for item in page.get("rec_texts", []):
            if isinstance(item, dict):
                texts.append({
                    "text": item.get("text", ""),
                    "confidence": item.get("score", 0),
                })
            elif isinstance(item, str):
                texts.append({"text": item, "confidence": 1.0})
    return texts

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # 默认测试：生成图片
        from PIL import Image, ImageDraw, ImageFont
        path = r"E:\soft\OCR\Paddle_ocr\test_images\test_chinese.png"
        img = Image.new('RGB', (600, 200), color='white')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 48)
        except Exception:
            font = ImageFont.load_default()
        draw.text((30, 20), "你好世界", fill='black', font=font)
        draw.text((30, 90), "PaddleOCR真棒!", fill='black', font=font)
        draw.text((30, 150), "RTX 4070 Ti SUPER", fill='black', font=font)
        img.save(path)
        print(f"测试图片已生成: {path}\n")

    texts = ocr_image(path)
    print("=" * 50)
    for t in texts:
        print(f"  文字: {t['text']:<20s}  置信度: {t['confidence']:.2%}")
    print("=" * 50)
    print(f"共识别 {len(texts)} 条文字")
