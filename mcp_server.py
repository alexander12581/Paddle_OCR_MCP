"""
PaddleOCR MCP Server
让 AI 能调用 PaddleOCR 识别图片中的文字。
模型在对话期间常驻内存，每次 OCR 后清理临时显存碎片。
启动方式: python mcp_server.py
"""
import os
import sys
import gc

# ============================================================
# 1. 环境配置（必须在 import paddleocr 之前）
# ============================================================

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

# ============================================================
# 2. 初始化 PaddleOCR（常驻内存）
# ============================================================

from paddleocr import PaddleOCR  # noqa: E402

_ocr = None

def _get_ocr():
    """懒加载模型，整个对话生命周期只加载一次"""
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(lang="ch", use_textline_orientation=False)
    return _ocr

def _cleanup_gpu():
    """释放本次 OCR 产生的临时显存碎片，不卸载模型"""
    try:
        import paddle
        paddle.device.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

# ============================================================
# 3. MCP Server
# ============================================================

from fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("paddleocr")


@mcp.tool
def recognize(image_path: str) -> dict:
    """识别图片中的所有文字。参数 image_path 是图片文件的完整路径。返回每行文字的内容和置信度。"""
    if not os.path.exists(image_path):
        return {"error": f"文件不存在: {image_path}"}

    ocr = _get_ocr()
    try:
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

        return {
            "image": os.path.basename(image_path),
            "text_count": len(texts),
            "texts": texts,
            "full_text": "\n".join(t["text"] for t in texts),
        }
    finally:
        _cleanup_gpu()


@mcp.tool
def ocr_status() -> dict:
    """检查 OCR 引擎状态。"""
    try:
        _get_ocr()
        return {"loaded": True, "language": "ch", "model_dir": _CACHE, "status": "ready"}
    except Exception as e:
        return {"loaded": False, "error": str(e), "status": "error"}


if __name__ == "__main__":
    sys.stderr.write("[paddleocr] Loading model...\n")
    sys.stderr.flush()
    # 启动时预加载模型（可注释掉改为懒加载）
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        _get_ocr()
    finally:
        sys.stdout = _real_stdout
    sys.stderr.write("[paddleocr] Model ready.\n")
    sys.stderr.flush()
    mcp.run()
