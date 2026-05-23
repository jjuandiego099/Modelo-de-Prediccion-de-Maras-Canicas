"""
Detección de Maras — FastAPI
API REST para inferencia con YOLOv8s
Autores: Juan Diego Chaparro García, Juan José Vargas, Santiago Amado
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
import time
import tempfile
import os
import subprocess
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from collections import Counter


# ── PostgreSQL ─────────────────────────────────────────────────────────────
import psycopg2

def get_db_connection():
    """
    Abre una conexión a PostgreSQL usando las mismas variables de entorno
    que usa app.py. El host 'postgres' corresponde al servicio en docker-compose.
    Retorna None (sin lanzar excepción) si la BD no está disponible.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "postgres"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "detecciones"),
            user=os.getenv("DB_USER", "admin"),
            password=os.getenv("DB_PASS", "canicas123"),
            connect_timeout=5,
        )
        return conn
    except Exception as e:
        print(f"⚠️  PostgreSQL no disponible: {e}")
        return None


def guardar_deteccion(fuente: str, detections: list, inference_ms: float) -> bool:
    """
    Persiste el resultado de una inferencia en la tabla 'detecciones'.
    Lógica idéntica a la de app.py — mismos nombres de columna y cálculos.

    Args:
        fuente:       "imagen", "video", "camara" o "movil"
        detections:   Lista de objetos Detection (con .class_name y .confidence)
        inference_ms: Tiempo de inferencia reportado por YOLOv8s

    Returns:
        True si se guardó correctamente, False si hubo error.
    """
    # Cuenta detecciones por clase (igual que app.py usa Counter)
    counts = Counter(d.class_name.lower() for d in detections)
    verde  = counts.get("green marble", 0)
    azul   = counts.get("blue marble",  0)
    blanca = counts.get("white marble", 0)
    negra  = counts.get("black marble", 0)
    total  = len(detections)

    # Confianza promedio — 0 si no hay detecciones
    conf_avg = round(
        float(np.mean([d.confidence for d in detections])), 4
    ) if detections else 0.0

    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO detecciones
                (fuente, verde, azul, blanca, negra, total, confianza_avg, inferencia_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (fuente, verde, azul, blanca, negra, total, conf_avg, round(inference_ms, 2)))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"⚠️  Error guardando en PostgreSQL: {e}")
        conn.close()
        return False


# ── Modelo Pydantic para /save (peticiones desde la app móvil) ─────────────
class SaveRequest(BaseModel):
    """
    Payload que envía la app móvil cuando ya tiene los conteos calculados
    (porque el móvil no puede llamar a guardar_deteccion directamente).
    Los nombres de campo son idénticos a las columnas de la tabla 'detecciones'.
    """
    fuente:        str   = "movil"
    verde:         int   = 0
    azul:          int   = 0
    blanca:        int   = 0
    negra:         int   = 0
    total:         int   = 0
    confianza_avg: float = 0.0
    inferencia_ms: float = 0.0


# ── Inicializar app ────────────────────────────────────────────────────────
app = FastAPI(
    title="API Detección de Maras",
    description="Detecta y clasifica 4 tipos de maras (canicas) usando YOLOv8s",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Carga del modelo ───────────────────────────────────────────────────────
model = None

def find_model() -> Optional[str]:
    base       = Path(__file__).parent
    candidates = [base / "best.pt", base / "weights" / "best.pt"]
    for c in candidates:
        if c.exists():
            return str(c)
    matches = sorted(base.glob("runs/train/*/weights/best.pt"), reverse=True)
    return str(matches[0]) if matches else None

@app.on_event("startup")
async def startup_event():
    global model
    from ultralytics import YOLO
    model_path = find_model()
    if not model_path:
        print("⚠️  No se encontró best.pt — colócalo junto a api.py")
        return
    model = YOLO(model_path)
    print(f"✅ Modelo cargado: {model_path}")
    print(f"   Clases: {model.names}")


# ── Modelos Pydantic ───────────────────────────────────────────────────────
class Detection(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    x1: int; y1: int; x2: int; y2: int
    width: int; height: int

class PredictResponse(BaseModel):
    success:          bool
    inference_ms:     float
    total_detections: int
    image_width:      int
    image_height:     int
    detections:       list[Detection]

class VideoStats(BaseModel):
    total_frames:             int
    processed_frames:         int
    total_detections:         int
    avg_detections_per_frame: float
    processing_ms:            float

class PredictVideoResponse(BaseModel):
    success: bool
    stats:   VideoStats


# ── Colores por clase (BGR) ────────────────────────────────────────────────
CLASS_COLORS = [
    (0, 229, 255),
    (255, 61, 87),
    (255, 214, 10),
    (160, 110, 255),
]


# ── Inferencia ─────────────────────────────────────────────────────────────
def run_inference(image_np: np.ndarray, conf: float = 0.35, iou: float = 0.45):
    results    = model.predict(source=image_np, conf=conf, iou=iou, imgsz=640, verbose=False)
    result     = results[0]
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id   = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf_val = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(Detection(
                class_id=cls_id, class_name=cls_name,
                confidence=round(conf_val, 4),
                x1=x1, y1=y1, x2=x2, y2=y2,
                width=x2 - x1, height=y2 - y1,
            ))
    return detections

def draw_boxes(image_np: np.ndarray, detections: list[Detection]) -> np.ndarray:
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
    return {
        "name": "API Detección de Maras",
        "version": "1.2.0",
        "model": "YOLOv8s",
        "classes": model.names if model else "Modelo no cargado",
        "endpoints": {
            "POST /predict":       "Recibe imagen → JSON con detecciones (guarda en BD)",
            "POST /predict/image": "Recibe imagen → PNG anotado",
            "POST /predict/video": "Recibe video  → MP4 anotado (guarda en BD)",
            "POST /save":          "Guarda conteos enviados por la app móvil",
            "GET  /health":        "Estado del servidor y modelo",
            "GET  /docs":          "Documentación Swagger",
        }
    }


@app.get("/health", tags=["Info"])
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {"status": "ok", "model_loaded": True, "classes": model.names, "num_classes": len(model.names)}


@app.post("/predict", response_model=PredictResponse, tags=["Inferencia"])
async def predict(
    file: UploadFile = File(..., description="Imagen JPG, PNG o BMP"),
    conf:   float = 0.5,
    iou:    float = 0.45,
    fuente: str   = "imagen",   # La app móvil pasa fuente=movil
):
    """
    Inferencia sobre imagen. Guarda el resultado en PostgreSQL automáticamente.
    El parámetro 'fuente' permite distinguir el origen: imagen | camara | movil.
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

    # Guardar en PostgreSQL con la misma lógica que app.py
    guardar_deteccion(fuente, detections, elapsed_ms)

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
    conf: float = 0.5,
    iou:  float = 0.45,
):
    """
    Retorna el PNG anotado. NO guarda en BD (el guardado ya lo hace /predict).
    Se llama en paralelo desde el frontend solo para obtener la imagen de calidad.
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
    _, buffer  = cv2.imencode(".png", annotated)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={"X-Detections": str(len(detections))},
    )


@app.post("/save", tags=["Base de datos"])
async def save(body: SaveRequest):
    """
    Endpoint exclusivo para la app móvil (Expo).
    Recibe los conteos ya calculados y los persiste directamente en PostgreSQL.
    Se usa cuando la app envía la imagen a /predict/image (solo para visualizar)
    y quiere guardar el resultado por separado con fuente='movil'.

    Nota: si la app llama a /predict (con fuente=movil), el guardado es automático
    y NO necesita llamar también a /save — haría un registro duplicado.
    """
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Base de datos no disponible")
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO detecciones
                (fuente, verde, azul, blanca, negra, total, confianza_avg, inferencia_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            body.fuente,
            body.verde,
            body.azul,
            body.blanca,
            body.negra,
            body.total,
            body.confianza_avg,
            body.inferencia_ms,
        ))
        conn.commit()
        cur.close()
        conn.close()
        return {"success": True, "message": "Detección guardada correctamente"}
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Error al guardar: {e}")


@app.post("/predict/video", tags=["Inferencia"])
async def predict_video(
    file:        UploadFile = File(..., description="Video MP4, AVI, MOV o MKV"),
    conf:        float = 0.5,
    iou:         float = 0.45,
    skip_frames: int   = 1,
    save:        bool  = True,
):
    """
    Inferencia sobre video completo. Por defecto guarda un registro de resumen en PostgreSQL
    al finalizar (con fuente='video' y las detecciones acumuladas de todos los frames).
    Pasa save=False para omitir el guardado en BD.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    allowed_types = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska", "video/x-msvideo"}
    if file.content_type not in allowed_types and not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="El archivo debe ser un video (MP4, AVI, MOV, MKV)")

    contents = await file.read()
    suffix   = Path(file.filename).suffix or ".mp4"
    in_tmp   = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    out_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    in_tmp.write(contents); in_tmp.flush(); in_tmp.close(); out_tmp.close()

    try:
        cap = cv2.VideoCapture(in_tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="No se pudo abrir el video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
        writer       = cv2.VideoWriter(out_tmp.name, fourcc, fps_vid, (w_vid, h_vid))

        frame_count       = 0
        processed         = 0
        all_detections    = []   # Acumula TODAS las detecciones del video para guardar en BD
        t_start           = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % skip_frames == 0:
                dets = run_inference(frame, conf=conf, iou=iou)
                writer.write(draw_boxes(frame, dets))
                all_detections.extend(dets)
                processed += 1
            else:
                writer.write(frame)

        cap.release()
        writer.release()
        elapsed_ms = (time.time() - t_start) * 1000

        # Guardar resumen del video en PostgreSQL (solo si save=True)
        if save:
            guardar_deteccion("video", all_detections, elapsed_ms)

        fixed_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fixed_tmp.close()
        subprocess.run([
            "ffmpeg", "-y", "-i", out_tmp.name,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            fixed_tmp.name
        ], check=True)

        with open(fixed_tmp.name, "rb") as f:
            video_bytes = f.read()
        os.unlink(fixed_tmp.name)

        return StreamingResponse(
            io.BytesIO(video_bytes),
            media_type="video/mp4",
            headers={
                "X-Total-Frames":     str(total_frames),
                "X-Processed-Frames": str(processed),
                "X-Total-Detections": str(len(all_detections)),
                "X-Processing-Ms":    str(round(elapsed_ms, 2)),
            },
        )
    finally:
        try: os.unlink(in_tmp.name)
        except Exception: pass
        try: os.unlink(out_tmp.name)
        except Exception: pass
def get_db():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "postgres"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "detecciones"),
        user=os.getenv("DB_USER", "admin"),
        password=os.getenv("DB_PASS", "canicas123"),
        options="-c timezone=America/Bogota"
    )

@app.get("/stats/totales", tags=["Stats"])
async def stats_totales():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COALESCE(SUM(verde),  0),
                COALESCE(SUM(azul),   0),
                COALESCE(SUM(blanca), 0),
                COALESCE(SUM(negra),  0)
            FROM detecciones
        """)
        row = cur.fetchone()
        return {
            "verde":  int(row[0]),
            "azul":   int(row[1]),
            "blanca": int(row[2]),
            "negra":  int(row[3]),
        }
    finally:
        cur.close()
        conn.close()


@app.get("/stats/historial", tags=["Stats"])
async def stats_historial(limit: int = 10, offset: int = 0):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM detecciones")
        total = cur.fetchone()[0]
        cur.execute("""
            SELECT id, fecha, fuente, verde, azul, blanca, negra, total
            FROM detecciones
            ORDER BY fecha DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        rows = cur.fetchall()
        return {
            "total": total,
            "rows": [
                {
                    "id":     r[0],
                    "fecha":  r[1].strftime("%m/%d %H:%M"),
                    "fuente": r[2],
                    "verde":  r[3],
                    "azul":   r[4],
                    "blanca": r[5],
                    "negra":  r[6],
                    "total":  r[7],
                }
                for r in rows
            ],
        }
    finally:
        cur.close()
        conn.close()