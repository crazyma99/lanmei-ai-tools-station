import io
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image

from . import exif_utils, fusion_core


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXIF_UPLOAD_DIR = DATA_DIR / "exif_uploads"
EXIF_PROCESSED_DIR = DATA_DIR / "exif_processed"
THUMBNAIL_DIR = DATA_DIR / "thumbnails"
PRESETS_DIR = PROJECT_ROOT / "presets"
FUSION_UPLOAD_DIR = DATA_DIR / "fusion_uploads"
FUSION_OUTPUT_DIR = DATA_DIR / "fusion_outputs"

for path in [
    EXIF_UPLOAD_DIR,
    EXIF_PROCESSED_DIR,
    THUMBNAIL_DIR,
    PRESETS_DIR,
    FUSION_UPLOAD_DIR,
    FUSION_OUTPUT_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_exif_storage_dirs():
    for folder in [EXIF_UPLOAD_DIR, EXIF_PROCESSED_DIR, THUMBNAIL_DIR]:
        if folder.is_dir():
            for name in os.listdir(folder):
                path = folder / name
                if path.is_file():
                    try:
                        path.unlink()
                    except Exception:
                        continue
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)


class ExifProcessRequest(BaseModel):
    id: str
    action: str
    convert_to_jpg: bool = False
    clear_aigc: bool = False
    add_noise: bool = False
    noise_intensity: int = 0
    preset: Optional[str] = None
    custom_data: Optional[Dict[str, Any]] = None
    deep_clean: bool = False


class BatchDownloadRequest(BaseModel):
    ids: List[str]


app = FastAPI(title="Lanmei AI Tools", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
def api_root():
    return {
        "status": "online",
        "message": "Lanmei AI Tools API Server",
        "version": "1.0.0",
    }


@app.post("/api/exif/upload")
async def upload_exif_file(file: UploadFile = File(...)):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="未选择文件")
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    file_id = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[1].lower()
    save_name = f"{file_id}.{ext}"
    upload_path = EXIF_UPLOAD_DIR / save_name
    content = await file.read()
    with upload_path.open("wb") as f:
        f.write(content)
    thumb_name = f"{file_id}_thumb.{ext}"
    thumb_path = THUMBNAIL_DIR / thumb_name
    exif_utils.create_thumbnail(str(upload_path), str(thumb_path))
    exif_data = exif_utils.get_exif_data(str(upload_path))
    aigc = exif_utils.detect_aigc_from_exif(exif_data)
    try:
        with Image.open(upload_path) as img:
            width, height = img.size
            fmt = img.format
    except Exception:
        width, height, fmt = None, None, None
    return {
        "id": file_id,
        "filename": file.filename,
        "thumbnail_url": f"/api/files/thumbnails/{thumb_name}",
        "exif": exif_data,
        "aigc": aigc.get("is_aigc", False),
        "aigc_detail": aigc,
        "width": width,
        "height": height,
        "format": fmt,
    }


def find_file_by_id(directory: Path, file_id: str, exclude_suffix: Optional[str] = None):
    if not directory.is_dir():
        return None
    for name in os.listdir(directory):
        if not name.startswith(file_id):
            continue
        if exclude_suffix and name.endswith(exclude_suffix):
            continue
        return directory / name
    return None


@app.post("/api/exif/process")
async def process_exif_file(payload: ExifProcessRequest):
    file_id = payload.id
    action = payload.action
    target_path = find_file_by_id(EXIF_UPLOAD_DIR, file_id)
    if target_path is None:
        raise HTTPException(status_code=404, detail="源文件不存在")
    convert_to_jpg = payload.convert_to_jpg
    clear_aigc = payload.clear_aigc
    add_noise = payload.add_noise
    noise_intensity = payload.noise_intensity
    if convert_to_jpg:
        output_filename = f"{file_id}.jpg"
    else:
        output_filename = target_path.name
    output_path = EXIF_PROCESSED_DIR / output_filename
    for name in os.listdir(EXIF_PROCESSED_DIR):
        if name.startswith(file_id) and name != output_filename:
            try:
                (EXIF_PROCESSED_DIR / name).unlink()
            except Exception:
                continue
    success = False
    if action == "deep_clean" or (action == "clear" and payload.deep_clean):
        success = exif_utils.deep_clean_image(
            str(target_path), str(output_path), intensity=noise_intensity
        )
    elif action == "clear":
        success = exif_utils.remove_exif(
            str(target_path),
            str(output_path),
            add_noise=add_noise,
            noise_intensity=noise_intensity,
        )
    elif action == "import_preset":
        if not payload.preset:
            raise HTTPException(status_code=400, detail="未指定预设名称")
        preset_path = PRESETS_DIR / f"{payload.preset}.json"
        if not preset_path.exists():
            raise HTTPException(status_code=404, detail="预设不存在")
        with preset_path.open("r", encoding="utf-8") as f:
            preset_data = json.load(f)
        success = exif_utils.modify_exif(
            str(target_path),
            str(output_path),
            preset_data=preset_data,
            convert_to_jpg=convert_to_jpg,
            add_noise=add_noise,
            noise_intensity=noise_intensity,
            deep_clean=payload.deep_clean,
        )
    elif action == "import_custom":
        if not payload.custom_data:
            raise HTTPException(status_code=400, detail="未提供自定义 EXIF 数据")
        success = exif_utils.modify_exif(
            str(target_path),
            str(output_path),
            preset_data=payload.custom_data,
            convert_to_jpg=convert_to_jpg,
            add_noise=add_noise,
            noise_intensity=noise_intensity,
            deep_clean=payload.deep_clean,
        )
    else:
        raise HTTPException(status_code=400, detail="不支持的处理动作")
    if not success:
        raise HTTPException(status_code=500, detail="处理失败")
    if payload.clear_aigc:
        try:
            exif_utils.strip_aigc_metadata(str(output_path), str(output_path))
        except Exception as e:
            print(f"strip_aigc_metadata error: {e}")
    new_exif = exif_utils.get_exif_data(str(output_path))
    new_aigc = exif_utils.detect_aigc_from_exif(new_exif)
    try:
        with Image.open(output_path) as img:
            n_width, n_height = img.size
            n_fmt = img.format
    except Exception:
        n_width, n_height, n_fmt = None, None, None
    return {
        "success": True,
        "exif": new_exif,
        "new_filename": output_filename if convert_to_jpg else None,
        "aigc": new_aigc.get("is_aigc", False),
        "aigc_detail": new_aigc,
        "width": n_width,
        "height": n_height,
        "format": n_fmt,
    }


@app.get("/api/exif/download/{file_id}")
async def download_exif_file(file_id: str):
    target_path = find_file_by_id(EXIF_PROCESSED_DIR, file_id)
    if target_path is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=str(target_path),
        filename=target_path.name,
        media_type="application/octet-stream",
    )


@app.get("/api/exif/view/upload/{file_id}")
async def view_exif_upload(file_id: str):
    target_path = None
    if EXIF_UPLOAD_DIR.is_dir():
        for name in os.listdir(EXIF_UPLOAD_DIR):
            if name.startswith(file_id) and "_thumb" not in name:
                target_path = EXIF_UPLOAD_DIR / name
                break
    if target_path is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path=str(target_path))


@app.get("/api/exif/view/output/{file_id}")
async def view_exif_output(file_id: str):
    target_path = None
    if EXIF_PROCESSED_DIR.is_dir():
        for name in os.listdir(EXIF_PROCESSED_DIR):
            if name.startswith(file_id) and not name.endswith(".zip"):
                target_path = EXIF_PROCESSED_DIR / name
                break
    if target_path is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path=str(target_path))


@app.post("/api/exif/download_batch")
async def download_batch(request: BatchDownloadRequest):
    if not request.ids:
        raise HTTPException(status_code=400, detail="未选择文件")
    zip_name = f"batch_download_{uuid.uuid4().hex}.zip"
    zip_path = EXIF_PROCESSED_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file_id in request.ids:
            target_path = find_file_by_id(EXIF_PROCESSED_DIR, file_id)
            if target_path is None:
                continue
            zipf.write(target_path, target_path.name)
    return FileResponse(
        path=str(zip_path),
        filename=zip_name,
        media_type="application/zip",
    )


@app.post("/api/session/cleanup")
async def session_cleanup():
    clean_exif_storage_dirs()
    for folder in [FUSION_UPLOAD_DIR, FUSION_OUTPUT_DIR]:
        if folder.is_dir():
            for name in os.listdir(folder):
                path = folder / name
                if path.is_file():
                    try:
                        path.unlink()
                    except Exception:
                        continue
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
    return {"success": True}


@app.get("/api/files/thumbnails/{filename}")
async def serve_thumbnails(filename: str):
    file_path = THUMBNAIL_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="缩略图不存在")
    return FileResponse(path=str(file_path))


@app.post("/api/fusion/process")
async def fusion_process(
    foreground: UploadFile = File(...),
    background: UploadFile = File(...),
    watermark: bool = Form(False),
):
    if foreground.filename == "" or background.filename == "":
        raise HTTPException(status_code=400, detail="请上传两张图片")
    fg_bytes = await foreground.read()
    bg_bytes = await background.read()
    try:
        img1_pil = Image.open(io.BytesIO(fg_bytes)).convert("RGB")
        img2_pil = Image.open(io.BytesIO(bg_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="图片格式无法解析")
    job_id = uuid.uuid4().hex
    job_dir = FUSION_OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    result = fusion_core.run_fusion(img1_pil, img2_pil, job_dir, add_watermark=watermark)
    variants = result.get("variants", [])
    if not variants:
        raise HTTPException(status_code=500, detail="叠图处理失败：无结果返回")
    
    # 优先选择成功的变体
    successful_variants = [v for v in variants if v["success"]]
    if not successful_variants:
        # 如果全部失败，返回第一个失败的原因
        error_msg = variants[0]["message"] if variants else "未知错误"
        raise HTTPException(status_code=500, detail=f"叠图失败: {error_msg}")
    
    # 默认选择第一个成功的变体 (通常是 Poisson Normal)
    best_variant = successful_variants[0]
    files = best_variant["urls"]
    wm_file = files.get('watermark')
    
    base = f"/api/files/fusion/{job_id}"
    return {
        "job_id": job_id,
        "message": best_variant["message"],
        "files": {
            "png_url": f"{base}/{files.get('png')}",
            "jpg_url": f"{base}/{files.get('jpg')}",
            "watermark_url": f"{base}/{wm_file}" if wm_file else None,
        },
        "variants": result["variants"], # 返回所有变体供前端扩展使用
    }


@app.get("/api/files/fusion/{job_id}/{filename}")
async def serve_fusion_file(job_id: str, filename: str):
    file_path = FUSION_OUTPUT_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path=str(file_path))


@app.post("/api/fusion/preview")
async def fusion_preview(file: UploadFile = File(...)):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="未选择文件")
    data = await file.read()
    try:
        img_pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="图片格式无法解析")
    preview = fusion_core.preview_segmentation(img_pil)
    if preview is None:
        raise HTTPException(status_code=400, detail="无法生成分割预览")
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
