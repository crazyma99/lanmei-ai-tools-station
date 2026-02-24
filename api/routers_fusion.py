from io import BytesIO
from typing import List
import uuid

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from PIL import Image
import numpy as np

from .fusion_core import run_fusion, DEVICE_WRAPPER
from .config import FUSION_OUTPUT_DIR


router = APIRouter(prefix="/fusion", tags=["fusion"])


@router.post("/process")
async def fusion_process(
    foreground: UploadFile = File(...),
    background: UploadFile = File(...),
    watermark: bool = Form(False),
):
    try:
        fg_bytes = await foreground.read()
        bg_bytes = await background.read()
        img1 = Image.open(BytesIO(fg_bytes)).convert("RGB")
        img2 = Image.open(BytesIO(bg_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="无法解析上传图片")

    def progress(_, __):
        return

    job_id = uuid.uuid4().hex
    job_dir = FUSION_OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    result = run_fusion(img1, img2, output_dir=job_dir, progress=progress, add_watermark=watermark)
    
    payload_variants = []
    base_url = f"/api/fusion/files/{job_id}"
    
    for item in result.get("variants", []):
        files = item.get("urls", {})
        # Ensure we have file names, not full URLs if fusion_core was changed
        wm_file = files.get('watermark')
        entry = {
            "title": item.get("title"),
            "success": item.get("success"),
            "message": item.get("message"),
            "resolution": item.get("resolution"),
            "urls": {
                "png_url": f"{base_url}/{files.get('png', '')}",
                "jpg_url": f"{base_url}/{files.get('jpg', '')}",
                "watermark_url": f"{base_url}/{wm_file}" if wm_file else None,
            } if item.get("success") else None,
        }
        payload_variants.append(entry)
        
    return {
        "job_id": job_id,
        "device": result.get("device", DEVICE_WRAPPER.label),
        "variants": payload_variants,
    }


@router.get("/files/{job_id}/{filename}")
async def download_fusion_image(job_id: str, filename: str):
    path = FUSION_OUTPUT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    # Determine media type based on extension
    media_type = "image/jpeg"
    if filename.lower().endswith(".png"):
        media_type = "image/png"
    return FileResponse(path=str(path), media_type=media_type)
