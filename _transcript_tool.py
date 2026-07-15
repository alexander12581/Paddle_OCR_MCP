# recognize_from_transcript tool body — injected into mcp_server.py
@mcp.tool
def recognize_from_transcript(transcript_path: str = "") -> dict:
    """
    从会话 transcript 中提取用户贴的图片并 OCR。
    transcript_path 可选，不传则自动在 ~/.claude/projects/ 下找最新 transcript。
    适用于多图粘贴（hook 只捕获最后一张）、拖文件贴图等场景。
    """
    import json as _json, base64 as _base64, uuid as _uuid, glob as _glob, tempfile as _tempfile
    from pathlib import Path as _Path

    # 自动发现 transcript
    if transcript_path and os.path.exists(transcript_path):
        tpath = transcript_path
    else:
        try:
            root = os.path.join(_Path.home(), ".claude", "projects")
            candidates = _glob.glob(os.path.join(root, "*", "*.jsonl"))
            if not candidates:
                return {"error": "未找到 transcript 文件"}
            tpath = max(candidates, key=os.path.getmtime)
        except Exception as e:
            return {"error": f"自动发现 transcript 失败: {e}"}

    if not os.path.exists(tpath):
        return {"error": f"文件不存在: {tpath}"}

    # 读 transcript，找最新含图片的 user 消息
    try:
        with open(tpath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception as e:
        return {"error": f"读取失败: {e}"}

    target_msg = None
    for line in reversed(lines):
        try:
            obj = _json.loads(line)
        except Exception:
            continue
        if obj.get("type") != "user": continue
        msg = obj.get("message") or {}
        if msg.get("role") != "user": continue
        content = msg.get("content")
        if not isinstance(content, list): continue
        if any(c.get("type") == "image" for c in content if isinstance(c, dict)):
            target_msg = obj; break

    if not target_msg:
        return {"error": "未找到含图片的 user 消息"}

    # 提取 base64 图片
    images = []
    for c in target_msg["message"]["content"]:
        if not isinstance(c, dict) or c.get("type") != "image": continue
        src = c.get("source") or {}
        if src.get("type") != "base64": continue
        data = src.get("data", "")
        if data:
            images.append({"media_type": src.get("media_type", "image/png"), "data": data})

    if not images:
        return {"error": "不含 base64 数据"}

    # 解码 + OCR
    results = []
    tmp_dir = _tempfile.gettempdir()
    for i, img in enumerate(images):
        try: raw = _base64.b64decode(img["data"])
        except Exception: results.append({"index": i, "error": "base64 解码失败"}); continue

        ext = ".jpg" if "jpeg" in img["media_type"] else ".png"
        fp = os.path.join(tmp_dir, f"ts_{_uuid.uuid4().hex[:8]}{ext}")
        try:
            with open(fp, "wb") as f: f.write(raw)
        except Exception as e:
            results.append({"index": i, "error": str(e)}); continue

        ocr = _get_ocr()
        try:
            ocr_result = ocr.predict(fp)
            texts = []
            for page in ocr_result:
                rt = page.get("rec_texts", []); rs = page.get("rec_scores", [])
                for j, item in enumerate(rt):
                    if isinstance(item, dict):
                        texts.append({"text": item.get("text", ""), "confidence": round(item.get("score", 0), 4)})
                    elif isinstance(item, str):
                        s = rs[j] if j < len(rs) else 1.0
                        texts.append({"text": item, "confidence": round(s, 4)})
            results.append({"index": i, "text_count": len(texts), "full_text": "\n".join(t["text"] for t in texts)})
        except Exception as e:
            results.append({"index": i, "error": str(e)})
        finally:
            _cleanup_gpu()
            try: os.unlink(fp)
            except Exception: pass

    return {"transcript": os.path.basename(tpath), "images_found": len(images), "results": results}
