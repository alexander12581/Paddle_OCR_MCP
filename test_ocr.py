"""
PaddleOCR PP-OCRv6 测试脚本
模型和缓存全部存到 E 盘
"""
import os
import sys

# === 修复 cuDNN 9.x 找不到 zlibwapi.dll（Windows 特有坑） ===
# zlibwapi.dll 已从 torch/lib 复制到 nvidia/cudnn/bin/
_ENV = r"E:\soft\anaconda3\envs\unlimited-ocr\Lib\site-packages"
os.add_dll_directory(os.path.join(_ENV, r"nvidia\cudnn\bin"))
os.add_dll_directory(os.path.join(_ENV, r"nvidia\cuda_runtime\bin"))
os.add_dll_directory(os.path.join(_ENV, r"nvidia\cublas\bin"))
os.add_dll_directory(os.path.join(_ENV, r"nvidia\cufft\bin"))
os.add_dll_directory(os.path.join(_ENV, r"nvidia\curand\bin"))
os.add_dll_directory(os.path.join(_ENV, r"nvidia\cusparse\bin"))
os.add_dll_directory(os.path.join(_ENV, r"nvidia\cusolver\bin"))
os.add_dll_directory(os.path.join(_ENV, r"torch\lib"))

# === 模型下载到 E 盘，不碰 C 盘 ===
PADDLE_HOME = r"E:\soft\OCR\Paddle_ocr\models"
os.environ["PADDLEX_HOME"] = PADDLE_HOME
os.environ["PADDLEOCR_HOME"] = PADDLE_HOME
os.makedirs(PADDLE_HOME, exist_ok=True)
print(f"模型/缓存目录: {PADDLE_HOME}")

from paddleocr import PaddleOCR

print("\n正在加载 PP-OCRv6 medium 模型...")
print("（首次运行会下载模型，约 30MB，请稍候）\n")

ocr = PaddleOCR(
    lang='ch',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print("\n模型加载成功！")

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
print("\nPP-OCRv6 测试通过！")
