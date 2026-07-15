"""关键测试：先import paddle（加载所有DLL），再import paddleocr"""
import os

# 加DLL目录
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

os.environ["PADDLEX_HOME"] = r"E:\soft\OCR\Paddle_ocr\models"

# 第一步：先直接 import paddle（让它初始化DLL，此时torch还没加载）
print("Step 1: import paddle (before torch)...")
import paddle
print(f"  GPU count: {paddle.device.cuda.device_count()}")

# 第二步：import torch
print("Step 2: import torch...")
import torch
print(f"  CUDA available: {torch.cuda.is_available()}")

# 第三步：import paddleocr
print("Step 3: import paddleocr...")
from paddleocr import PaddleOCR
print("  OK!")

# 第四步：创建OCR对象
print("Step 4: Create OCR pipeline...")
ocr = PaddleOCR(lang="ch", use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False)
print("SUCCESS!")
