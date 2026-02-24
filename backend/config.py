from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
THUMBNAIL_DIR = DATA_DIR / "thumbnails"
FUSION_OUTPUT_DIR = DATA_DIR / "fusion_outputs"

MODELS_DIR = BASE_DIR / "models"
BISE_NET_DIR = MODELS_DIR / "bisenet"
BISE_ONNX_PATH = BISE_NET_DIR / "resnet18.onnx"
BISE_MODEL_PATH = BISE_NET_DIR / "bisenet_face.pth"


def ensure_directories() -> None:
    for path in [DATA_DIR, UPLOAD_DIR, PROCESSED_DIR, THUMBNAIL_DIR, FUSION_OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
