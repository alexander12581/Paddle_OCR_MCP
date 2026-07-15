# PaddleOCR MCP Server

让不支持图片的纯文本模型（DeepSeek V4 等）也能"看图"的 MCP Server。

## 原理

PaddleOCR MCP Server 为 AI 提供 OCR 文字识别能力。AI 调用 `recognize` 工具传入图片路径，返回图中所有文字内容。

## 安装

```bash
# 1. 创建 conda 环境
conda create -n paddle_ocr python=3.10
conda activate paddle_ocr

# 2. 安装依赖
pip install paddlepaddle-gpu paddleocr fastmcp

# 3. 配置 Claude Code
# 在项目根目录的 .mcp.json 中添加：
{
  "mcpServers": {
    "paddleocr": {
      "command": "YOUR_PYTHON_PATH/python.exe",
      "args": ["PATH_TO/mcp_server.py"]
    }
  }
}
```

## 硬件要求

|          | 最低                 | 推荐                    |
|----------|---------------------|------------------------|
| GPU      | NVIDIA GTX 1050 4GB | NVIDIA RTX 3060+ 8GB   |
| VRAM     | 4 GB                | 8 GB+                  |
| RAM      | 4 GB                | 8 GB+                  |
| CUDA     | 11.2+               | 12.x                   |
| cuDNN    | 8.x                 | 9.x                    |
| 磁盘     | ~500MB              | SSD                    |

无 GPU 时自动退到 CPU 模式，速度慢 5-10 倍但仍可用。

## 工具

### recognize

识别图片中的所有文字。

- **参数**: `image_path` — 图片文件的完整路径
- **返回**: 每行文字的内容和置信度

### ocr_status

检查 OCR 引擎状态。

## 配合 image-detect Hook 实现自动贴图识别

本项目推荐配合 [image-detect hook](https://github.com/amen-xu/paddleocr-mcp) 使用，实现"用户 Ctrl+V 贴图 → AI 自动识别"的丝滑体验。

详见 [hooks/](./hooks/) 目录。

## 架构

```
用户贴图 → hook 读剪贴板 → 保存 Temp → PENDING_IMAGE 注入 AI 上下文
  → AI 调 mcp__paddleocr__recognize → PaddleOCR GPU/CPU 推理 → 返回文字
```

## License

MIT
