from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi import Body
from pydantic import BaseModel
from starlette.responses import FileResponse

from .config import UPLOAD_DIR, PROCESSED_DIR, THUMBNAIL_DIR
from .exif_utils import (
    create_thumbnail,
    get_exif_data,
    detect_aigc_from_exif,
    deep_clean_image,
    remove_exif,
    modify_exif,
    strip_aigc_metadata,
)

import os
import uuid


router = APIRouter(prefix="/exif", tags=["exif"])


class ExifProcessRequest(BaseModel):
    id: str
    action: str
    convert_to_jpg: bool = False
    clear_aigc: bool = False
    add_noise: bool = False
    noise_intensity: int = 0
    deep_clean: bool = False
    preset: Optional[str] = None
    custom_data: Optional[dict] = None


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext.replace(".", "") not in {"png", "jpg", "jpeg", "tiff", "webp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    file_id = str(uuid.uuid4())
    save_name = f"{file_id}{ext}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / save_name
    with upload_path.open("wb") as f:
        content = await file.read()
        f.write(content)
    thumb_name = f"{file_id}_thumb{ext}"
    thumb_path = THUMBNAIL_DIR / thumb_name
    create_thumbnail(str(upload_path), str(thumb_path))
    exif_data = get_exif_data(str(upload_path))
    aigc = detect_aigc_from_exif(exif_data)
    width = height = None
    fmt = None
    try:
        from PIL import Image

        with Image.open(upload_path) as img:
            width, height = img.size
            fmt = img.format
    except Exception:
        pass
    return {
        "id": file_id,
        "filename": file.filename,
        "thumbnail_url": f"/static/thumbnails/{thumb_name}",
        "exif": exif_data,
        "aigc": aigc.get("is_aigc", False),
        "aigc_detail": aigc,
        "width": width,
        "height": height,
        "format": fmt,
    }


@router.post("/process")
async def process_image(payload: ExifProcessRequest = Body(...)):
    file_id = payload.id
    files = [p for p in UPLOAD_DIR.glob(f"{file_id}.*") if not p.name.endswith("_thumb")]
    if not files:
        raise HTTPException(status_code=404, detail="File not found")
    input_path = files[0]
    convert_to_jpg = payload.convert_to_jpg
    clear_aigc = payload.clear_aigc
    add_noise = payload.add_noise
    noise_intensity = payload.noise_intensity
    if convert_to_jpg:
        output_filename = f"{file_id}.jpg"
    else:
        output_filename = input_path.name
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for p in PROCESSED_DIR.glob(f"{file_id}*"):
        if p.name != output_filename:
            try:
                p.unlink()
            except Exception:
                pass
    output_path = PROCESSED_DIR / output_filename
    success = False
    if payload.action == "deep_clean" or (payload.action == "clear" and payload.deep_clean):
        success = deep_clean_image(str(input_path), str(output_path), intensity=noise_intensity)
    elif payload.action == "clear":
        success = remove_exif(
            str(input_path),
            str(output_path),
            add_noise=add_noise,
            noise_intensity=noise_intensity,
        )
    elif payload.action == "import_custom":
        if payload.custom_data:
            success = modify_exif(
                str(input_path),
                str(output_path),
                preset_data=payload.custom_data,
                convert_to_jpg=convert_to_jpg,
                add_noise=add_noise,
                noise_intensity=noise_intensity,
                deep_clean=payload.deep_clean,
            )
        else:
            raise HTTPException(status_code=400, detail="No custom data provided")
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    if not success:
        raise HTTPException(status_code=500, detail="Processing failed")
    if clear_aigc:
        try:
            strip_aigc_metadata(str(output_path), str(output_path))
        except Exception as e:
            print(f"strip_aigc_metadata error: {e}")
    new_exif = get_exif_data(str(output_path))
    new_aigc = detect_aigc_from_exif(new_exif)
    width = height = None
    fmt = None
    try:
        from PIL import Image

        with Image.open(output_path) as img:
            width, height = img.size
            fmt = img.format
    except Exception:
        pass
    return {
        "success": True,
        "exif": new_exif,
        "new_filename": output_filename if convert_to_jpg else None,
        "aigc": new_aigc.get("is_aigc", False),
        "aigc_detail": new_aigc,
        "width": width,
        "height": height,
        "format": fmt,
        "download_url": f"/static/processed/{output_filename}",
    }


@router.get("/download/{file_id}")
async def download_processed(file_id: str):
    files = [p for p in PROCESSED_DIR.glob(f"{file_id}*") if not p.name.endswith(".zip")]
    if not files:
        raise HTTPException(status_code=404, detail="File not found")
    path = files[0]
    return FileResponse(path)

