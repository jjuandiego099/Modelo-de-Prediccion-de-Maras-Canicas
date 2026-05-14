"""
Detección de Maras — FastAPI
API REST para inferencia con YOLOv8s
Autores: Juan Diego Chaparro García, Juan José Vargas, Santiago Amado
"""

# Importaciones de FastAPI y utilidades HTTP
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware          # Permite peticiones desde otros orígenes (ej. el frontend Streamlit)
from fastapi.responses import JSONResponse, StreamingResponse  # Tipos de respuesta HTTP
import numpy as np          # Operaciones numéricas con arrays (necesario para manejar imágenes)
import cv2                  # OpenCV: lectura, decodificación y dibujo sobre imágenes/video
import io                   # Manejo de flujos de bytes en memoria (evita escribir archivos temporales)
import time                 # Medición de tiempos de inferencia
import tempfile             # Archivos temporales en disco para videos
import os                   # Manejo del sistema de archivos
import subprocess           # Ejecutar comandos externos (ffmpeg para re-encodear video)
from pathlib import Path    # Rutas de archivos de forma multiplataforma
from typing import Optional # Tipo de dato para valores opcionales
from pydantic import BaseModel  # Validación automática de esquemas de datos


# ── Inicializar app ────────────────────────────────────────────────────────
# Se crea la instancia principal de FastAPI con metadatos de documentación
app = FastAPI(
    title="API Detección de Maras",
    description="Detecta y clasifica 4 tipos de maras (canicas) usando YOLOv8s",
    version="1.1.0",
    docs_url="/docs",       # Swagger UI disponible en /docs
    redoc_url="/redoc",     # ReDoc disponible en /redoc
)


# ── CORS ───────────────────────────────────────────────────────────────────
# Middleware que permite que cualquier origen (frontend, Postman, etc.) llame a la API
# En producción se recomienda restringir allow_origins a dominios específicos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Acepta peticiones de cualquier dominio
    allow_methods=["*"],   # Acepta cualquier método HTTP (GET, POST, etc.)
    allow_headers=["*"],   # Acepta cualquier cabecera HTTP
)


# ── Carga del modelo ───────────────────────────────────────────────────────
# Variable global que almacenará la instancia del modelo YOLO una vez cargado
model = None


def find_model() -> Optional[str]:
    """
    Busca el archivo de pesos 'best.pt' del modelo YOLOv8s en ubicaciones predefinidas.
    Primero busca en la raíz del proyecto y en la carpeta 'weights/',
    luego busca en los resultados de entrenamiento de Ultralytics (runs/train/).
    Retorna la ruta al primer archivo encontrado, o None si no existe.
    """
    base = Path(__file__).parent  # Directorio donde está este script

    # Rutas candidatas donde podría estar el modelo
    candidates = [
        base / "best.pt",
        base / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # Busca en carpetas de entrenamiento de Ultralytics, ordenando de más reciente a más antiguo
    matches = sorted(base.glob("runs/train/*/weights/best.pt"), reverse=True)
    if matches:
        return str(matches[0])

    return None  # No se encontró ningún modelo


@app.on_event("startup")
async def startup_event():
    """
    Evento que se ejecuta automáticamente al iniciar el servidor.
    Carga el modelo YOLOv8s desde disco y lo deja listo en memoria para inferencia.
    Si no se encuentra el archivo best.pt, el servidor arranca igual pero sin modelo.
    """
    global model
    from ultralytics import YOLO  # Se importa aquí para no ralentizar el arranque si falla

    model_path = find_model()
    if not model_path:
        print("⚠️  No se encontró best.pt — colócalo junto a api.py")
        return

    model = YOLO(model_path)  # Carga el modelo en memoria
    print(f"✅ Modelo cargado: {model_path}")
    print(f"   Clases: {model.names}")  # Muestra el diccionario id→nombre de clases


# ── Modelos de respuesta ───────────────────────────────────────────────────
# Esquemas Pydantic que definen y validan la estructura de los datos de entrada/salida

class Detection(BaseModel):
    """Representa una sola detección: clase, confianza y coordenadas del bounding box."""
    class_id: int       # Índice numérico de la clase (0–3 para las 4 tipos de maras)
    class_name: str     # Nombre legible de la clase (ej. "green marble")
    confidence: float   # Probabilidad de la detección (0.0 – 1.0)
    x1: int             # Coordenada x del borde izquierdo del bounding box
    y1: int             # Coordenada y del borde superior del bounding box
    x2: int             # Coordenada x del borde derecho del bounding box
    y2: int             # Coordenada y del borde inferior del bounding box
    width: int          # Ancho del bounding box en píxeles (x2 - x1)
    height: int         # Alto del bounding box en píxeles (y2 - y1)


class PredictResponse(BaseModel):
    """Respuesta completa del endpoint /predict para imágenes."""
    success: bool               # Indica si la inferencia fue exitosa
    inference_ms: float         # Tiempo de inferencia en milisegundos
    total_detections: int       # Número total de objetos detectados
    image_width: int            # Ancho de la imagen procesada en píxeles
    image_height: int           # Alto de la imagen procesada en píxeles
    detections: list[Detection] # Lista de todas las detecciones individuales


class VideoStats(BaseModel):
    """Estadísticas de procesamiento de un video."""
    total_frames: int                   # Frames totales del video original
    processed_frames: int               # Frames a los que se aplicó inferencia
    total_detections: int               # Detecciones acumuladas en todo el video
    avg_detections_per_frame: float     # Promedio de detecciones por frame procesado
    processing_ms: float                # Tiempo total de procesamiento en milisegundos


class PredictVideoResponse(BaseModel):
    """Respuesta del endpoint /predict/video con estadísticas del procesamiento."""
    success: bool
    stats: VideoStats


# ── Colores por clase ──────────────────────────────────────────────────────
# Colores en formato BGR (no RGB) para dibujar los bounding boxes con OpenCV.
# El orden corresponde a las 4 clases del modelo.
CLASS_COLORS = [
    (0, 229, 255),    # Clase 0: Cian/Azul claro
    (255, 61, 87),    # Clase 1: Rojo/Rosa
    (255, 214, 10),   # Clase 2: Amarillo
    (160, 110, 255),  # Clase 3: Púrpura
]


# ── Inferencia ─────────────────────────────────────────────────────────────

def run_inference(image_np: np.ndarray, conf: float = 0.35, iou: float = 0.45):
    """
    Ejecuta el modelo YOLOv8s sobre un array NumPy (imagen BGR) y retorna
    la lista de detecciones encontradas.

    Args:
        image_np: Imagen de entrada como array NumPy en formato BGR.
        conf: Umbral de confianza mínima. Detecciones por debajo se descartan.
        iou: Umbral de IoU para Non-Maximum Suppression (elimina cajas duplicadas).

    Returns:
        Lista de objetos Detection con la información de cada objeto detectado.
    """
    results = model.predict(
        source=image_np,    # Imagen de entrada
        conf=conf,          # Umbral de confianza
        iou=iou,            # Umbral IoU para NMS
        imgsz=640,          # Tamaño estándar de entrada del modelo YOLOv8s
        verbose=False,      # Suprime el log por consola de cada predicción
    )
    result     = results[0]  # results es una lista; tomamos el primer (y único) resultado
    detections = []

    # Itera sobre cada caja detectada
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id   = int(box.cls.item())              # ID numérico de la clase
            cls_name = model.names[cls_id]              # Nombre de la clase
            conf_val = float(box.conf.item())           # Confianza de esta detección
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Coordenadas del bbox

            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=round(conf_val, 4),
                x1=x1, y1=y1, x2=x2, y2=y2,
                width=x2 - x1,   # Ancho calculado a partir de las coordenadas
                height=y2 - y1,  # Alto calculado a partir de las coordenadas
            ))

    return detections


def draw_boxes(image_np: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """
    Dibuja los bounding boxes y etiquetas sobre una copia de la imagen.

    Por cada detección:
      1. Dibuja un rectángulo del color de la clase.
      2. Dibuja un fondo sólido para la etiqueta (para mejor legibilidad).
      3. Escribe el nombre de la clase y la confianza sobre el fondo.

    Args:
        image_np: Imagen original en formato BGR (no se modifica).
        detections: Lista de detecciones a dibujar.

    Returns:
        Nueva imagen (copia) con los bounding boxes y etiquetas dibujados.
    """
    annotated = image_np.copy()  # Trabaja sobre una copia para no modificar el original

    for det in detections:
        color = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]  # Color según la clase

        # Dibuja el rectángulo del bounding box (grosor 2px)
        cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        # Texto de la etiqueta: "nombre_clase confianza"
        label = f"{det.class_name} {det.confidence:.2f}"

        # Mide el tamaño del texto para ajustar el fondo
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)

        # Dibuja el rectángulo de fondo de la etiqueta (relleno sólido con -1)
        cv2.rectangle(annotated, (det.x1, det.y1 - th - 10), (det.x1 + tw + 8, det.y1), color, -1)

        # Escribe el texto sobre el fondo
        cv2.putText(annotated, label, (det.x1 + 4, det.y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (10, 12, 15), 1, cv2.LINE_AA)

    return annotated


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """
    Endpoint raíz informativo.
    Retorna el nombre, versión, modelo activo y lista de endpoints disponibles.
    """
    return {
        "name": "API Detección de Maras",
        "version": "1.1.0",
        "model": "YOLOv8s",
        "classes": model.names if model else "Modelo no cargado",
        "endpoints": {
            "POST /predict":       "Recibe imagen → JSON con detecciones",
            "POST /predict/image": "Recibe imagen → PNG anotado",
            "POST /predict/video": "Recibe video → MP4 anotado",
            "GET  /health":        "Estado del servidor y modelo",
            "GET  /docs":          "Documentación Swagger",
        }
    }


@app.get("/health", tags=["Info"])
async def health():
    """
    Endpoint de salud del servicio.
    Retorna 503 si el modelo no está cargado, o 200 con información del modelo.
    Útil para healthchecks de Docker/Kubernetes y para el frontend Streamlit.
    """
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
    conf: float = 0.5,   # Umbral de confianza (parámetro de query string)
    iou:  float = 0.45,  # Umbral IoU para NMS (parámetro de query string)
):
    """
    Endpoint principal de inferencia sobre imágenes.
    Recibe una imagen, aplica YOLOv8s y retorna las detecciones en formato JSON.

    - Valida que el archivo sea una imagen (por content_type).
    - Decodifica los bytes con OpenCV.
    - Ejecuta inferencia y mide el tiempo.
    - Retorna un PredictResponse con todas las detecciones.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    # Validación del tipo de archivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    # Lectura y decodificación de la imagen desde bytes a array NumPy
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decodifica en BGR

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    h, w = img_bgr.shape[:2]  # Dimensiones de la imagen

    # Inferencia con medición de tiempo
    t0         = time.time()
    detections = run_inference(img_bgr, conf=conf, iou=iou)
    elapsed_ms = (time.time() - t0) * 1000  # Convertir segundos a milisegundos

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
    Endpoint de inferencia que retorna la imagen anotada (con bounding boxes dibujados).
    La respuesta es un PNG en flujo de bytes (StreamingResponse), no un JSON.
    El header 'X-Detections' informa cuántos objetos fueron detectados.
    Útil para previsualizar resultados directamente en el navegador o el frontend.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    # Decodificación de la imagen
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    # Inferencia y dibujo de boxes
    detections = run_inference(img_bgr, conf=conf, iou=iou)
    annotated  = draw_boxes(img_bgr, detections)

    # Codificación de la imagen anotada a PNG en memoria
    _, buffer = cv2.imencode(".png", annotated)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),       # Flujo de bytes del PNG
        media_type="image/png",
        headers={"X-Detections": str(len(detections))},  # Número de detecciones en el header
    )


@app.post("/predict/video", tags=["Inferencia"])
async def predict_video(
    file: UploadFile = File(..., description="Video MP4, AVI, MOV o MKV"),
    conf:        float = 0.5,
    iou:         float = 0.45,
    skip_frames: int   = 1,    # Procesar 1 de cada N frames (1 = todos los frames)
):
    """
    Endpoint de inferencia sobre videos completos.
    Flujo de trabajo:
      1. Guarda el video recibido en un archivo temporal de entrada.
      2. Procesa frame a frame con run_inference y draw_boxes.
      3. Escribe el video anotado en un archivo temporal de salida (codec mp4v).
      4. Re-encodea con ffmpeg a H.264/yuv420p para compatibilidad con navegadores.
      5. Retorna el video final como StreamingResponse con estadísticas en headers.
      6. Limpia todos los archivos temporales al finalizar (bloque finally).

    El parámetro skip_frames permite omitir frames para acelerar el procesamiento:
      skip_frames=1 → inferencia en cada frame
      skip_frames=2 → inferencia en 1 de cada 2 frames (los omitidos se copian tal cual)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    # Validación del tipo de archivo de video
    allowed_types = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska", "video/x-msvideo"}
    if file.content_type not in allowed_types and not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="El archivo debe ser un video (MP4, AVI, MOV, MKV)")

    contents = await file.read()
    suffix   = Path(file.filename).suffix or ".mp4"  # Extensión del archivo original

    # Archivos temporales: in_tmp para el video de entrada, out_tmp para el de salida
    in_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    in_tmp.write(contents)  # Escribe el video recibido en disco
    in_tmp.flush()
    in_tmp.close()
    out_tmp.close()

    try:
        # Abre el video con OpenCV para leer sus propiedades y frames
        cap = cv2.VideoCapture(in_tmp.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="No se pudo abrir el video")

        # Propiedades del video original
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 25.0  # FPS del video (25 por defecto si no se puede leer)
        w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Escritor de video de salida con codec mp4v (compatible con OpenCV)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_tmp.name, fourcc, fps_vid, (w_vid, h_vid))

        # Contadores para las estadísticas finales
        frame_count = 0
        total_dets  = 0
        processed   = 0
        t_start     = time.time()

        # Procesamiento frame a frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Fin del video o error de lectura

            frame_count += 1

            # Aplicar inferencia solo en los frames que correspondan según skip_frames
            if frame_count % skip_frames == 0:
                dets = run_inference(frame, conf=conf, iou=iou)
                annotated_frame = draw_boxes(frame, dets)
                writer.write(annotated_frame)   # Escribe el frame anotado
                total_dets += len(dets)
                processed  += 1
            else:
                writer.write(frame)  # Escribe el frame sin anotar (frames omitidos)

        cap.release()
        writer.release()

        elapsed_ms = (time.time() - t_start) * 1000  # Tiempo total en milisegundos

        # Re-encodear a H.264 con ffmpeg para garantizar compatibilidad con navegadores
        # mp4v no es reproducible directamente en Chrome/Firefox; H.264 + yuv420p sí lo es
        fixed_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fixed_tmp.close()
        subprocess.run([
            "ffmpeg", "-y",          # -y: sobreescribe sin preguntar
            "-i", out_tmp.name,      # Video de entrada (codec mp4v)
            "-vcodec", "libx264",    # Re-encodear a H.264
            "-pix_fmt", "yuv420p",   # Formato de píxel compatible con todos los navegadores
            fixed_tmp.name           # Video de salida final
        ], check=True)  # check=True lanza excepción si ffmpeg falla

        # Lee el video final y lo devuelve como flujo de bytes
        with open(fixed_tmp.name, "rb") as f:
            video_bytes = f.read()

        os.unlink(fixed_tmp.name)  # Elimina el archivo temporal de ffmpeg

        # Retorna el video con estadísticas en headers HTTP personalizados
        return StreamingResponse(
            io.BytesIO(video_bytes),
            media_type="video/mp4",
            headers={
                "X-Total-Frames":     str(total_frames),           # Frames totales del video
                "X-Processed-Frames": str(processed),              # Frames con inferencia
                "X-Total-Detections": str(total_dets),             # Detecciones totales
                "X-Processing-Ms":    str(round(elapsed_ms, 2)),   # Tiempo de procesamiento
            },
        )

    finally:
        # Limpieza garantizada de archivos temporales, incluso si hubo errores
        try:
            os.unlink(in_tmp.name)
        except Exception:
            pass
        try:
            os.unlink(out_tmp.name)
        except Exception:
            pass    