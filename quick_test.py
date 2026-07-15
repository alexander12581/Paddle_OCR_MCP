"""PaddleOCR PP-OCRv6 快速测试"""
import os

# DLL 修复
_ENV = r"E:\soft\anaconda3\envs\unlimited-ocr\Lib\site-packages"
for d in [
    r"nvidia\cudnn\bin",
    r"nvidia\cublas\bin",
    r"nvidia\cufft\bin",
    r"nvidia\curand\bin",
    r"nvidia\cusparse\bin",
    r"nvidia\cusolver\bin",
    r"torch\lib",
]:
    os.add_dll_directory(os.path.join(_ENV, d))

# 模型存 E 盘
os.environ["PADDLEX_HOME"] = r"E:\soft\OCR\Paddle_ocr\models"

# 关键：先 import torch 再 import paddleocr
import torch  # noqa
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="ch")
print("OK!")
