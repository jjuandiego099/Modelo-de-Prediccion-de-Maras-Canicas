"""
Detección de Maras — FastAPI
API REST para inferencia con YOLOv8s
Autores: Juan Diego Chaparro García, Juan José Vargas, Santiago Amado
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import cv2
import io
import time
import tempfile
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

# ── Inicializar app ────────────────────────────────────────────────────────
app = FastAPI(
    title="API Detección de Maras",
    description="Detecta y clasifica 4 tipos de maras (canicas) usando YOLOv8s",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — permite requests desde el frontend Streamlit ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Carga del modelo al arrancar ───────────────────────────────────────────
model = None

def find_model() -> Optional[str]:
    """Busca best.pt automáticamente en la carpeta del proyecto."""
    base = Path(__file__).parent
    candidates = [
        base / "best.pt",
        base / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    matches = sorted(base.glob("runs/train/*/weights/best.pt"), reverse=True)
    if matches:
        return str(matches[0])
    return None


@app.on_event("startup")
async def startup_event():
    """Carga el modelo YOLOv8 al iniciar la API."""
    global model
    from ultralytics import YOLO
    model_path = find_model()
    if not model_path:
        print("⚠️  No se encontró best.pt — colócalo junto a api.py")
        return
    model = YOLO(model_path)
    print(f"✅ Modelo cargado: {model_path}")
    print(f"   Clases: {model.names}")


# ── Modelos de respuesta ───────────────────────────────────────────────────
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int


class PredictResponse(BaseModel):
    success: bool
    inference_ms: float
    total_detections: int
    image_width: int
    image_height: int
    detections: list[Detection]


class VideoStats(BaseModel):
    total_frames: int
    processed_frames: int
    total_detections: int
    avg_detections_per_frame: float
    processing_ms: float


class PredictVideoResponse(BaseModel):
    success: bool
    stats: VideoStats


# ── Colores por clase ──────────────────────────────────────────────────────
CLASS_COLORS = [
    (0, 229, 255),
    (255, 61, 87),
    (255, 214, 10),
    (160, 110, 255),
]


# ── Función de inferencia ──────────────────────────────────────────────────
def run_inference(image_np: np.ndarray, conf: float = 0.35, iou: float = 0.45):
    """Corre YOLOv8 sobre un array numpy BGR. Retorna lista de Detection."""
    results = model.predict(
        source=image_np,
        conf=conf,
        iou=iou,
        imgsz=640,
        verbose=False,
    )
    result     = results[0]
    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id   = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf_val = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=round(conf_val, 4),
                x1=x1, y1=y1, x2=x2, y2=y2,
                width=x2 - x1,
                height=y2 - y1,
            ))

    return detections


def draw_boxes(image_np: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Dibuja bounding boxes sobre imagen BGR."""
    annotated = image_np.copy()
    for det in detections:
        color = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]
        cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(annotated, (det.x1, det.y1 - th - 10), (det.x1 + tw + 8, det.y1), color, -1)
        cv2.putText(annotated, label, (det.x1 + 4, det.y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (10, 12, 15), 1, cv2.LINE_AA)
    return annotated


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """Info general de la API."""
    return {
        "name": "API Detección de Maras",
        "version": "1.1.0",
        "model": "YOLOv8s",
        "classes": model.names if model else "Modelo no cargado",
        "endpoints": {
            "POST /predict":        "Recibe imagen → JSON con detecciones",
            "POST /predict/image":  "Recibe imagen → PNG anotado",
            "POST /predict/video":  "Recibe video → MP4 anotado",
            "GET  /health":         "Estado del servidor y modelo",
            "GET  /docs":           "Documentación Swagger",
        }
    }


@app.get("/health", tags=["Info"])
async def health():
    """Verifica que el servidor y el modelo estén listos."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {
        "status": "ok",
        "model_loaded": True,
        "classes": model.names,
        "num_classes": len(model.names),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inferencia"])
async def predict(
    file: UploadFile = File(..., description="Imagen JPG, PNG o BMP"),
    conf: float = 0.35,
    iou:  float = 0.45,
):
    """
    Recibe una imagen y devuelve las detecciones en formato JSON.

    - **file**: imagen a analizar (JPG, PNG, BMP, WEBP)
    - **conf**: umbral de confianza (0.1 – 1.0)
    - **iou**: umbral IoU para NMS (0.1 – 1.0)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    h, w = img_bgr.shape[:2]

    t0         = time.time()
    detections = run_inference(img_bgr, conf=conf, iou=iou)
    elapsed_ms = (time.time() - t0) * 1000

    return PredictResponse(
        success=True,
        inference_ms=round(elapsed_ms, 2),
        total_detections=len(detections),
        image_width=w,
        image_height=h,
        detections=detections,
    )


@app.post("/predict/image", tags=["Inferencia"])
async def predict_image(
    file: UploadFile = File(..., description="Imagen JPG, PNG o BMP"),
    conf: float = 0.35,
    iou:  float = 0.45,
):
    """
    Recibe una imagen y devuelve la misma imagen con los bounding boxes dibujados (PNG).

    - **file**: imagen a analizar
    - **conf**: umbral de confianza
    - **iou**: umbral IoU para NMS
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    detections = run_inference(img_bgr, conf=conf, iou=iou)
    annotated  = draw_boxes(img_bgr, detections)

    _, buffer = cv2.imencode(".png", annotated)
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={"X-Detections": str(len(detections))},
    )


@app.post("/predict/video", tags=["Inferencia"])
async def predict_video(
    file: UploadFile = File(..., description="Video MP4, AVI, MOV o MKV"),
    conf:        float = 0.35,
    iou:         float = 0.45,
    skip_frames: int   = 1,
):
    """
    Recibe un video y devuelve el mismo video con bounding boxes dibujados (MP4).

    - **file**: video a analizar
    - **conf**: umbral de confianza (0.1 – 1.0)
    - **iou**: umbral IoU para NMS (0.1 – 1.0)
    - **skip_frames**: procesar 1 de cada N frames (1 = todos). Los frames omitidos
      se copian sin anotaciones para mantener la duración original del video.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    allowed_types = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska", "video/x-msvideo"}
    if file.content_type not in allowed_types and not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="El archivo debe ser un video (MP4, AVI, MOV, MKV)")

    # Guardar el video recibido en un temporal
    contents = await file.read()
    suffix   = Path(file.filename).suffix or ".mp4"

    in_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    in_tmp.write(contents)
    in_tmp.flush()
    in_tmp.close()
    out_tmp.close()

    try:
        cap = cv2.VideoCapture(in_tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="No se pudo abrir el video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_tmp.name, fourcc, fps_vid, (w_vid, h_vid))

        frame_count  = 0
        total_dets   = 0
        processed    = 0
        t_start      = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % skip_frames == 0:
                dets = run_inference(frame, conf=conf, iou=iou)
                annotated_frame = draw_boxes(frame, dets)
                writer.write(annotated_frame)
                total_dets += len(dets)
                processed  += 1
            else:
                # Frame omitido — escribir sin anotaciones para mantener duración
                writer.write(frame)

        cap.release()
        writer.release()

        elapsed_ms = (time.time() - t_start) * 1000

        # Leer el video resultante y devolverlo como streaming
        with open(out_tmp.name, "rb") as f:
            video_bytes = f.read()

        return StreamingResponse(
            io.BytesIO(video_bytes),
            media_type="video/mp4",
            headers={
                "X-Total-Frames":     str(total_frames),
                "X-Processed-Frames": str(processed),
                "X-Total-Detections": str(total_dets),
                "X-Processing-Ms":    str(round(elapsed_ms, 2)),
            },
        )

    finally:
        # Limpiar temporales siempre, incluso si hay error
        try:
            os.unlink(in_tmp.name)
        except Exception:
            pass
        try:
            os.unlink(out_tmp.name)
        except Exception:
            pass