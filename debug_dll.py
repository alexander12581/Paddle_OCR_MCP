import ctypes, os

dll = r"E:\soft\anaconda3\envs\unlimited-ocr\Lib\site-packages\nvidia\cudnn\bin\cudnn_cnn64_9.dll"
zlib = r"E:\soft\anaconda3\envs\unlimited-ocr\Lib\site-packages\nvidia\cudnn\bin\zlibwapi.dll"

print("cudnn exists:", os.path.exists(dll))
print("zlib exists:", os.path.exists(zlib))
if os.path.exists(zlib):
    print("zlib size:", os.path.getsize(zlib))

# Try to load zlib first
try:
    h = ctypes.WinDLL(zlib)
    print("zlibwapi loaded OK, handle:", h)
except Exception as e:
    print("zlibwapi load error:", e)

# Then try cudnn
try:
    h = ctypes.WinDLL(dll)
    print("cudnn_cnn loaded OK, handle:", h)
except Exception as e:
    print("cudnn_cnn load error:", e)
