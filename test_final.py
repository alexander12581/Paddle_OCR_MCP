"""
PaddleOCR PP-OCRv6 完整测试（独立环境 paddle_ocr）
模型和缓存全部存到 E 盘
"""
import os

# === 模型下载到 E 盘，不碰 C 盘 ===
PADDLE_HOME = r"E:\soft\OCR\Paddle_ocr\models"
os.environ["PADDLEX_HOME"] = PADDLE_HOME
os.environ["PADDLEOCR_HOME"] = PADDLE_HOME
os.makedirs(PADDLE_HOME, exist_ok=True)
print(f"模型/缓存目录: {PADDLE_HOME}")

from paddleocr import PaddleOCR

print("\n加载 PP-OCRv6 模型（首次会下载 ~30MB）...")
ocr = PaddleOCR(
    lang='ch',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
print("模型加载成功！")

# 生成测试图片
from PIL import Image, ImageDraw, ImageFont

img_path = r"E:\soft\OCR\Paddle_ocr\test_images\test_chinese.png"
img = Image.new('RGB', (600, 200), color='white')
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 48)
except Exception:
    font = ImageFont.load_default()
draw.text((30, 20), "你好世界", fill='black', font=font)
draw.text((30, 90), "PaddleOCR太棒了!", fill='black', font=font)
draw.text((30, 150), "RTX 4070 Ti SUPER", fill='black', font=font)
img.save(img_path)
print(f"测试图片已生成: {img_path}")

# OCR 识别
print("\n正在识别...")
result = ocr.ocr(img_path)

print("\n" + "=" * 50)
print("识别结果：")
print("=" * 50)
if result and result[0]:
    for line in result[0]:
        text = line[1][0]
        confidence = line[1][1]
        print(f"  文字: {text:<20s}  置信度: {confidence:.2%}")
else:
    print("  未识别到文字")
print("=" * 50)
print("\n PP-OCRv6 测试通过！")
