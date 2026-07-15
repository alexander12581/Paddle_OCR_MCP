"""
OCR 单张图片，打印原始返回结果
用法: python ocr_raw.py <图片路径>
"""
import os
import sys
import json
import numpy as np
from pprint import pprint

os.environ["PADDLE_PDX_CACHE_HOME"] = r"E:\soft\OCR\Paddle_ocr\models"

from paddleocr import PaddleOCR

# ====== 可调参数 ======
LANG = "ch"           # 语言：ch/en/...
SHOW_POLYS = 10        # 坐标框显示几个（-1=全部）
SHOW_TEXTS = 20        # 文字显示几条（-1=全部）
PRETTY_PRINT = True    # True=pprint缩进 / False=JSON一行
# =====================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def simplify_page(page, max_polys=10, max_texts=20):
    """精简输出，坐标/文字只展示前 N 条"""
    p = {}
    for k, v in page.items():
        if k in ("dt_polys", "rec_polys"):
            arr = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
            p[k] = arr[:max_polys] if max_polys >= 0 else arr
            if max_polys >= 0 and len(arr) > max_polys:
                p[f"{k}_truncated"] = f"只显示前 {max_polys} / 共 {len(arr)}"
        elif isinstance(v, np.ndarray):
            p[k] = v.tolist()[:max_texts] if max_texts >= 0 else v.tolist()
            if max_texts >= 0 and len(v) > max_texts:
                p[f"{k}_truncated"] = f"只显示前 {max_texts} / 共 {len(v)}"
        elif isinstance(v, list) and k not in ("rec_texts", "rec_scores", "vis_fonts", "textline_orientation_angles"):
            p[k] = v[:max_texts] if max_texts >= 0 else v
        elif isinstance(v, dict):
            # doc_preprocessor_res 特殊处理：转可序列化
            clean = {}
            for dk, dv in v.items():
                if isinstance(dv, np.ndarray):
                    clean[dk] = f"ndarray shape={dv.shape} dtype={dv.dtype}"
                elif callable(dv) or "weakref" in str(type(dv)):
                    clean[dk] = str(type(dv).__name__)
                else:
                    try:
                        json.dumps(dv)
                        clean[dk] = dv
                    except TypeError:
                        clean[dk] = str(type(dv).__name__)
            p[k] = clean
        else:
            p[k] = v
    return p

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python ocr_raw.py <图片路径>")
        print("示例: python ocr_raw.py test.png")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"文件不存在: {img_path}")
        sys.exit(1)

    print(f"加载模型 (lang={LANG}) ...")
    ocr = PaddleOCR(lang=LANG)

    print(f"识别: {img_path}")
    result = ocr.predict(img_path)

    print(f"\n返回 {len(result)} 页\n")
    print("=" * 60)

    for i, page in enumerate(result):
        print(f"\n{'='*60}")
        print(f"第 {i+1} 页 (共 {len(page)} 个字段)")
        print(f"{'='*60}\n")

        simplified = simplify_page(page, max_polys=SHOW_POLYS, max_texts=SHOW_TEXTS)

        if PRETTY_PRINT:
            pprint(simplified, width=120, sort_dicts=False)
        else:
            print(json.dumps(simplified, cls=NumpyEncoder, indent=2, ensure_ascii=False))
