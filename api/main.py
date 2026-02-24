from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import ensure_directories, THUMBNAIL_DIR, PROCESSED_DIR, FUSION_OUTPUT_DIR, DATA_DIR
from .routers_exif import router as exif_router
from .routers_fusion import router as fusion_router


ensure_directories()

app = FastAPI(title="Lanmei AI Tools API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(exif_router, prefix="/api")
app.include_router(fusion_router, prefix="/api")


app.mount(
    "/static/thumbnails",
    StaticFiles(directory=str(THUMBNAIL_DIR)),
    name="thumbnails",
)
app.mount(
    "/static/processed",
    StaticFiles(directory=str(PROCESSED_DIR)),
    name="processed",
)
app.mount(
    "/static/fusion",
    StaticFiles(directory=str(FUSION_OUTPUT_DIR)),
    name="fusion",
)
app.mount(
    "/static",
    StaticFiles(directory=str(DATA_DIR)),
    name="static",
)


@app.get("/api")
def api_root():
    return {
        "status": "online",
        "message": "Lanmei AI Tools API",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)

