"""
Detección de Maras — Streamlit App
Proyecto: Detección de 4 tipos de maras usando YOLOv8s
Autores: Juan Diego Chaparro García, Juan José Vargas, Santiago Amado

Modelo: best.pt (YOLOv8s fine-tuned)
  → Se prefiere best.pt sobre ONNX porque:
    • Ultralytics lo carga nativamente sin dependencias extra
    • Soporta GPU automáticamente si está disponible
    • Misma velocidad en CPU para este tamaño de modelo (small)
    • ONNX requeriría onnxruntime + post-procesamiento manual de outputs
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from PIL import Image

# ── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Detección de Maras",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Variables de color */
:root {
    --bg:       #0a0c0f;
    --surface:  #111418;
    --border:   #1e2530;
    --accent:   #00e5ff;
    --accent2:  #ff3d57;
    --accent3:  #ffd60a;
    --text:     #e8edf2;
    --muted:    #64748b;
}

/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Header principal */
.main-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -1px;
    color: var(--text);
    margin: 0;
}
.main-header h1 span { color: var(--accent); }
.main-header .subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.main-header .authors {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.75rem;
}

/* Tarjetas de métricas */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.25rem;
}

/* Badges de clases detectadas */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.15rem;
}
.badge-0 { background: rgba(0,229,255,0.15); color: #00e5ff; border: 1px solid rgba(0,229,255,0.3); }
.badge-1 { background: rgba(255,61,87,0.15);  color: #ff3d57; border: 1px solid rgba(255,61,87,0.3); }
.badge-2 { background: rgba(255,214,10,0.15); color: #ffd60a; border: 1px solid rgba(255,214,10,0.3); }
.badge-3 { background: rgba(160,110,255,0.15);color: #a06eff; border: 1px solid rgba(160,110,255,0.3); }

/* Sección de resultados */
.results-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    margin-top: 1rem;
}
.results-box h3 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin: 0 0 0.75rem 0;
    font-weight: 600;
}

/* Info box modelo */
.model-info {
    background: rgba(0,229,255,0.05);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent);
    margin-bottom: 1rem;
}

/* Botones Streamlit */
[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
}

/* Slider */
[data-testid="stSlider"] .stSlider > div { color: var(--accent) !important; }

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Divider con estilo */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}

/* Spinner / estado */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
}
.status-ok   { background: rgba(0,229,255,0.1); color: var(--accent); border: 1px solid rgba(0,229,255,0.3); }
.status-warn { background: rgba(255,214,10,0.1); color: var(--accent3); border: 1px solid rgba(255,214,10,0.3); }
.status-err  { background: rgba(255,61,87,0.1);  color: var(--accent2); border: 1px solid rgba(255,61,87,0.3); }

/* Ocultar elementos por defecto de Streamlit */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Carga del modelo (cacheado) ────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    """
    Carga el modelo YOLOv8 desde best.pt.
    Se prefiere best.pt sobre ONNX porque:
      - Carga nativa con Ultralytics sin dependencias adicionales
      - Gestión automática de GPU/CPU
      - Para YOLOv8s en CPU, la diferencia de velocidad con ONNX es mínima (~5%)
      - ONNX requeriría post-procesamiento manual de outputs (NMS, decoding)
    """
    from ultralytics import YOLO
    return YOLO(model_path)


# ── Colores por clase (RGBA para OpenCV) ───────────────────────────────────
CLASS_COLORS = [
    (0, 229, 255),    # Clase 0 — cyan
    (255, 61, 87),    # Clase 1 — rojo
    (255, 214, 10),   # Clase 2 — amarillo
    (160, 110, 255),  # Clase 3 — violeta
]
BADGE_CLASSES = ["badge-0", "badge-1", "badge-2", "badge-3"]


def run_inference(model, image_np: np.ndarray, conf: float, iou: float):
    """
    Ejecuta la inferencia sobre un frame numpy (BGR o RGB).
    Retorna: imagen anotada (RGB), lista de detecciones [{class, name, conf, box}]
    """
    results = model.predict(
        source=image_np,
        conf=conf,
        iou=iou,
        imgsz=640,
        verbose=False,
    )
    result = results[0]

    # Dibujar bounding boxes manualmente para control total del estilo
    annotated = image_np.copy()
    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id   = int(box.cls.item())
            cls_name = model.names[cls_id] if model.names else f"Clase {cls_id}"
            conf_val = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

            # Rectángulo principal
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Etiqueta con fondo
            label = f"{cls_name}  {conf_val:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                annotated, label,
                (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.55,
                (10, 12, 15), 1, cv2.LINE_AA,
            )

            detections.append({
                "class_id": cls_id,
                "name": cls_name,
                "conf": conf_val,
                "box": (x1, y1, x2, y2),
            })

    # Convertir BGR→RGB si la imagen viene de OpenCV
    if len(annotated.shape) == 3 and annotated.shape[2] == 3:
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    else:
        annotated_rgb = annotated

    return annotated_rgb, detections


def detections_summary(detections: list, model_names: dict) -> str:
    """Genera HTML con badges de resumen de detecciones."""
    if not detections:
        return '<span style="color:#64748b; font-size:0.85rem;">Sin detecciones</span>'
    from collections import Counter
    counts = Counter(d["name"] for d in detections)
    parts = []
    for i, (name, count) in enumerate(counts.items()):
        cls_ids = [d["class_id"] for d in detections if d["name"] == name]
        badge_cls = BADGE_CLASSES[cls_ids[0] % 4] if cls_ids else "badge-0"
        parts.append(f'<span class="badge {badge_cls}">{name} × {count}</span>')
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════
#  LAYOUT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
  <h1>🔮 Detección de <span>Maras</span></h1>
  <p class="subtitle">Sistema de Visión por Computador · YOLOv8s</p>
  <p class="authors">Juan Diego Chaparro García &nbsp;·&nbsp; Juan José Vargas &nbsp;·&nbsp; Santiago Amado</p>
</div>
""", unsafe_allow_html=True)

# ── Sección explicativa del proyecto ──────────────────────────────────────
with st.expander("📖 ¿Qué es este proyecto y cómo funciona?", expanded=True):
    col_desc, col_clases = st.columns([3, 2])

    with col_desc:
        st.markdown("""
<div style="font-family:sans-serif; line-height:1.8;">
<h4 style="color:#00e5ff; margin-top:0; font-size:1rem; text-transform:uppercase; letter-spacing:2px;">¿Qué son las maras?</h4>
<p style="color:#e8edf2; font-size:0.93rem;">
En la región de Santander (Colombia), las <strong style="color:#00e5ff;">maras</strong> son las <strong>canicas</strong>,
pequeñas esferas de vidrio de colores con las que se juega desde hace generaciones.
Este proyecto aplica <strong>Inteligencia Artificial</strong> para detectar y clasificar automáticamente
4 tipos de maras según su color, usando una cámara o imágenes y videos.
</p>
<h4 style="color:#00e5ff; font-size:1rem; text-transform:uppercase; letter-spacing:2px;">¿Cómo funciona?</h4>
<p style="color:#e8edf2; font-size:0.93rem;">
El sistema usa <strong>YOLOv8s</strong> (You Only Look Once), un modelo de detección de objetos
en tiempo real entrenado específicamente con imágenes de maras. El proceso es:
</p>
<ol style="color:#e8edf2; font-size:0.93rem; line-height:2.2;">
  <li>📷 <strong>Captura</strong> — Se toma una imagen, video o frame de cámara</li>
  <li>🧠 <strong>Inferencia</strong> — YOLOv8s analiza cada región de la imagen buscando maras</li>
  <li>📦 <strong>Detección</strong> — Dibuja un bounding box alrededor de cada mara encontrada</li>
  <li>🎨 <strong>Clasificación</strong> — Identifica el color de cada mara con su porcentaje de confianza</li>
</ol>
<p style="color:#64748b; font-size:0.82rem; font-family:monospace;">
El modelo fue entrenado con ~530 imágenes anotadas en Roboflow y desplegado en AWS EC2.
Se eligió YOLOv8s (small) por su equilibrio entre precisión y eficiencia en CPU.
</p>
</div>
""", unsafe_allow_html=True)

    with col_clases:
        st.markdown("""
<div style="font-family:sans-serif;">
<h4 style="color:#00e5ff; margin-top:0; font-size:1rem; text-transform:uppercase; letter-spacing:2px;">Clases detectadas</h4>
<div style="display:flex; flex-direction:column; gap:0.75rem; margin-top:1rem;">

  <div style="background:#111418; border:1px solid #1e2530; border-left:4px solid #00c853; border-radius:6px; padding:0.75rem 1rem; display:flex; align-items:center; gap:1rem;">
    <span style="font-size:1.6rem;">🟢</span>
    <div>
      <div style="color:#00c853; font-weight:700; font-size:0.95rem;">Mara Verde</div>
      <div style="color:#64748b; font-size:0.78rem; font-family:monospace;">green marble · 124 muestras</div>
    </div>
  </div>

  <div style="background:#111418; border:1px solid #1e2530; border-left:4px solid #2979ff; border-radius:6px; padding:0.75rem 1rem; display:flex; align-items:center; gap:1rem;">
    <span style="font-size:1.6rem;">🔵</span>
    <div>
      <div style="color:#2979ff; font-weight:700; font-size:0.95rem;">Mara Azul</div>
      <div style="color:#64748b; font-size:0.78rem; font-family:monospace;">blue marble · 147 muestras</div>
    </div>
  </div>

  <div style="background:#111418; border:1px solid #1e2530; border-left:4px solid #e0e0e0; border-radius:6px; padding:0.75rem 1rem; display:flex; align-items:center; gap:1rem;">
    <span style="font-size:1.6rem;">⚪</span>
    <div>
      <div style="color:#e0e0e0; font-weight:700; font-size:0.95rem;">Mara Blanca</div>
      <div style="color:#64748b; font-size:0.78rem; font-family:monospace;">white marble · 144 muestras</div>
    </div>
  </div>

  <div style="background:#111418; border:1px solid #1e2530; border-left:4px solid #ffd60a; border-radius:6px; padding:0.75rem 1rem; display:flex; align-items:center; gap:1rem;">
    <span style="font-size:1.6rem;">⚫</span>
    <div>
      <div style="color:#ffd60a; font-weight:700; font-size:0.95rem;">Mara Negra</div>
      <div style="color:#64748b; font-size:0.78rem; font-family:monospace;">black marble · 116 muestras</div>
    </div>
  </div>

</div>
</div>
""", unsafe_allow_html=True)

# ── Auto-detección del modelo en la carpeta del proyecto ──────────────────
def find_model() -> str | None:
    """
    Busca best.pt de forma automática en la carpeta del proyecto.
    Orden de búsqueda:
      1. Mismo directorio que app.py
      2. Subcarpeta weights/ (típica de YOLOv8)
      3. Cualquier runs/train/*/weights/best.pt generado por Ultralytics
      4. Cualquier *.pt en el árbol del proyecto (primer resultado)
    """
    base = Path(__file__).parent  # carpeta donde vive app.py

    # 1. Junto a app.py
    candidate = base / "best.pt"
    if candidate.exists():
        return str(candidate)

    # 2. weights/best.pt
    candidate = base / "weights" / "best.pt"
    if candidate.exists():
        return str(candidate)

    # 3. runs/train/.../weights/best.pt  (salida típica de model.train())
    matches = sorted(base.glob("runs/train/*/weights/best.pt"), reverse=True)
    if matches:
        return str(matches[0])  # la carpeta más reciente primero

    # 4. Cualquier .pt en el árbol (excluyendo entornos virtuales)
    for pt in base.rglob("*.pt"):
        parts = pt.parts
        if any(p in parts for p in ("venv", ".venv", "env", "__pycache__", "site-packages")):
            continue
        return str(pt)

    return None  # no encontrado


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    # Buscar modelo automáticamente
    auto_path = find_model()

    if auto_path:
        model_path = auto_path
        rel = Path(auto_path).relative_to(Path(__file__).parent) if Path(auto_path).is_relative_to(Path(__file__).parent) else Path(auto_path)
        st.markdown(
            f'<div class="model-info">✓ Modelo detectado automáticamente<br>'
            f'📁 <code>{rel}</code></div>',
            unsafe_allow_html=True,
        )
    else:
        model_path = None
        st.markdown(
            '<div class="status-err" style="margin-bottom:0.75rem;">❌ No se encontró <code>best.pt</code> en el proyecto.<br>'
            'Asegúrate de que el archivo esté en la misma carpeta que <code>app.py</code>.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="model-info">📦 Formato: YOLOv8s · best.pt<br>⚡ Backend: PyTorch (CPU/GPU auto)</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Umbral de confianza ────────────────────────────────────────────────
    st.markdown("**🎯 Umbral de confianza**")
    conf_threshold = st.slider("conf", 0.1, 1.0, 0.35, 0.05, label_visibility="collapsed")
    with st.expander("ℹ️ ¿Qué es?"):
        st.markdown(
            '<div style="font-size:0.82rem;color:#94a3b8;line-height:1.7;">'
            'Porcentaje mínimo de certeza para reportar una detección.<br><br>'
            '<b style="color:#00e5ff;">Valor alto (0.7–0.9):</b> Menos detecciones, más precisas. Reduce falsos positivos.<br>'
            '<b style="color:#ffd60a;">Valor bajo (0.2–0.4):</b> Más detecciones, puede incluir objetos que no son maras.<br><br>'
            '<i>Recomendado: 0.50–0.65 para este modelo.</i>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Umbral IoU ─────────────────────────────────────────────────────────
    st.markdown("**📐 Umbral IoU (NMS)**")
    iou_threshold = st.slider("iou", 0.1, 1.0, 0.45, 0.05, label_visibility="collapsed")
    with st.expander("ℹ️ ¿Qué es?"):
        st.markdown(
            '<div style="font-size:0.82rem;color:#94a3b8;line-height:1.7;">'
            '<b style="color:#e8edf2;">IoU</b> = <i>Intersection over Union</i>. Controla el algoritmo '
            '<b>NMS</b> que elimina cajas duplicadas cuando el modelo detecta el mismo objeto varias veces.<br><br>'
            '<b style="color:#00e5ff;">Valor alto (0.6–0.9):</b> Permite más cajas superpuestas. '
            'Útil si las maras están muy juntas.<br>'
            '<b style="color:#ffd60a;">Valor bajo (0.2–0.4):</b> Elimina más duplicados, pero puede '
            'perder maras cercanas entre sí.<br><br>'
            '<i>Recomendado: 0.40–0.50 para objetos pequeños como canicas.</i>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Cámara ─────────────────────────────────────────────────────────────
    st.markdown("**📹 Cámara**")
    camera_index = st.number_input("Índice de cámara", min_value=0, max_value=4, value=0, step=1)
    camera_fps   = st.slider("FPS objetivo", 1, 30, 10, 1)
    with st.expander("ℹ️ ¿Qué es?"):
        st.markdown(
            '<div style="font-size:0.82rem;color:#94a3b8;line-height:1.7;">'
            '<b style="color:#e8edf2;">Índice de cámara:</b> número que identifica qué cámara usar. '
            '<span style="font-family:monospace;color:#00e5ff;">0</span> = webcam principal, '
            '<span style="font-family:monospace;color:#00e5ff;">1</span> = segunda cámara conectada.<br><br>'
            '<b style="color:#e8edf2;">FPS objetivo:</b> frames por segundo que se intentan procesar. '
            'Mayor FPS = más fluidez, pero más carga en CPU.<br><br>'
            '<b style="color:#ffd60a;">💡 Nota:</b> el stream en vivo usa <b>WebRTC</b> directamente '
            'desde el navegador, por lo que el índice solo aplica en modo local con OpenCV.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; color:#64748b; text-align:center; font-family:\'JetBrains Mono\',monospace;">Detección · 4 clases de maras<br>Modelo desplegado en AWS EC2</p>', unsafe_allow_html=True)

# ── Carga del modelo ───────────────────────────────────────────────────────
model = None

if not model_path:
    st.markdown(
        '<div class="status-err">❌ No se encontró <code>best.pt</code>. '
        'Coloca el archivo en la misma carpeta que <code>app.py</code> y recarga la página.</div>',
        unsafe_allow_html=True,
    )
else:
    with st.spinner("Cargando modelo YOLOv8s..."):
        try:
            model = load_model(model_path)
            num_classes = len(model.names) if model.names else "?"
            rel = Path(model_path).name
            st.markdown(
                f'<div class="status-ok">✓ Modelo listo &nbsp;|&nbsp; {num_classes} clases &nbsp;|&nbsp; <code>{rel}</code></div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown(f'<div class="status-err">❌ Error al cargar el modelo: {e}</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Tabs principales ───────────────────────────────────────────────────────
tab_img, tab_vid, tab_cam = st.tabs(["📷  Imagen", "🎬  Video", "📹  Cámara en vivo"])


# ════════════════════════════════════════
#  TAB 1 — IMAGEN
# ════════════════════════════════════════
with tab_img:
    st.markdown("#### Subir imagen")
    uploaded_img = st.file_uploader(
        "Formatos soportados: JPG, JPEG, PNG, BMP, WEBP",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_uploader",
        label_visibility="collapsed",
    )

    if uploaded_img and model:
        # Leer imagen
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col_orig, col_pred = st.columns(2)

        with col_orig:
            st.markdown('<div class="results-box"><h3>Original</h3>', unsafe_allow_html=True)
            st.image(img_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Inferencia
        with st.spinner("Detectando..."):
            t0 = time.time()
            annotated, detections = run_inference(model, img_bgr, conf_threshold, iou_threshold)
            elapsed = time.time() - t0

        with col_pred:
            st.markdown('<div class="results-box"><h3>Detecciones</h3>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Métricas
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="val">{len(detections)}</div><div class="lbl">Detecciones</div></div>', unsafe_allow_html=True)
        with m2:
            avg_conf = np.mean([d["conf"] for d in detections]) if detections else 0
            st.markdown(f'<div class="metric-card"><div class="val">{avg_conf:.2f}</div><div class="lbl">Conf. media</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="val">{elapsed*1000:.0f}ms</div><div class="lbl">Inferencia</div></div>', unsafe_allow_html=True)
        with m4:
            h, w = img_bgr.shape[:2]
            st.markdown(f'<div class="metric-card"><div class="val">{w}×{h}</div><div class="lbl">Resolución</div></div>', unsafe_allow_html=True)

        # Clases detectadas
        st.markdown(
            f'<div class="results-box" style="margin-top:1rem;"><h3>Clases encontradas</h3>{detections_summary(detections, model.names)}</div>',
            unsafe_allow_html=True,
        )

        # Tabla de detecciones
        if detections:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            with st.expander("📋 Tabla de detecciones"):
                import pandas as pd
                rows = []
                for i, d in enumerate(detections):
                    x1, y1, x2, y2 = d["box"]
                    rows.append({
                        "#": i + 1,
                        "Clase": d["name"],
                        "Confianza": f"{d['conf']:.4f}",
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "Ancho": x2 - x1, "Alto": y2 - y1,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Descarga de imagen resultante
        result_pil = Image.fromarray(annotated)
        import io
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️  Descargar imagen con detecciones",
            data=buf.getvalue(),
            file_name=f"deteccion_{uploaded_img.name}",
            mime="image/png",
        )

    elif uploaded_img and not model:
        st.warning("⚠️ Carga el modelo primero desde la barra lateral.")


# ════════════════════════════════════════
#  TAB 2 — VIDEO
# ════════════════════════════════════════
with tab_vid:
    st.markdown("#### Subir video")
    uploaded_vid = st.file_uploader(
        "Formatos soportados: MP4, AVI, MOV, MKV",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_uploader",
        label_visibility="collapsed",
    )

    if uploaded_vid and model:
        import io as _io
        from collections import Counter

        # Guardar video en archivo temporal para OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_vid.name).suffix)
        tfile.write(uploaded_vid.read())
        tfile.flush()
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 25
        w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration     = total_frames / fps_vid if fps_vid > 0 else 0
        cap.release()

        st.markdown(
            f'<div class="model-info">🎬 {uploaded_vid.name} &nbsp;|&nbsp; '
            f'{total_frames} frames &nbsp;|&nbsp; {fps_vid:.1f} FPS &nbsp;|&nbsp; '
            f'{w_vid}×{h_vid} &nbsp;|&nbsp; {duration:.1f}s</div>',
            unsafe_allow_html=True,
        )

        # ── Preview del video original ────────────────────────────────────
        st.markdown('<div class="results-box"><h3>Video original</h3>', unsafe_allow_html=True)
        st.video(tfile.name)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Opciones ──────────────────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            skip_frames = st.slider("Procesar 1 de cada N frames", 1, 10, 1, 1,
                                    help="1 = todos los frames (más lento, mejor resultado)")
        with col_b:
            max_frames_proc = st.slider("Máx. frames a procesar", 50, 1000, 300, 50)

        if st.button("▶️  Procesar video con detección", use_container_width=True):

            # Reabrir captura
            cap = cv2.VideoCapture(tfile.name)

            # Archivo de salida temporal (MP4 con codec H264)
            out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_path = out_file.name
            out_file.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps_vid, (w_vid, h_vid))

            # Placeholders para previsualización lado a lado
            st.markdown("##### Procesando...")
            col_orig, col_det = st.columns(2)
            with col_orig:
                st.markdown('<div class="results-box"><h3>Original</h3>', unsafe_allow_html=True)
                ph_orig = st.empty()
                st.markdown('</div>', unsafe_allow_html=True)
            with col_det:
                st.markdown('<div class="results-box"><h3>Detecciones</h3>', unsafe_allow_html=True)
                ph_det = st.empty()
                st.markdown('</div>', unsafe_allow_html=True)

            progress_bar      = st.progress(0)
            stats_placeholder = st.empty()

            frame_count = 0
            total_dets  = 0
            all_classes = []
            processed   = 0
            t_start     = time.time()

            while cap.isOpened() and processed < max_frames_proc:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                # Siempre escribir frame al video de salida (con o sin inferencia)
                if frame_count % skip_frames == 0:
                    # Inferencia
                    annotated_f, dets = run_inference(model, frame, conf_threshold, iou_threshold)
                    total_dets  += len(dets)
                    all_classes.extend([d["name"] for d in dets])
                    processed   += 1

                    # Escribir frame anotado al video de salida (BGR)
                    out_bgr = cv2.cvtColor(annotated_f, cv2.COLOR_RGB2BGR)
                    writer.write(out_bgr)

                    # Previsualización — mostrar cada 5 frames procesados para no saturar UI
                    if processed % 5 == 0:
                        orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ph_orig.image(orig_rgb, use_container_width=True)
                        ph_det.image(annotated_f, use_container_width=True, channels="RGB")
                else:
                    # Frame no procesado — escribir original al video
                    writer.write(frame)

                # Barra de progreso
                pct = min(processed / max_frames_proc, 1.0)
                progress_bar.progress(pct)

                # Stats
                elapsed_t = time.time() - t_start
                fps_proc  = processed / elapsed_t if elapsed_t > 0 else 0
                stats_placeholder.markdown(
                    f'<div class="results-box">'
                    f'<h3>Procesando...</h3>'
                    f'🎞️ Frame {frame_count}/{total_frames} &nbsp;|&nbsp; '
                    f'✅ Procesados: {processed} &nbsp;|&nbsp; '
                    f'🔍 Detecciones: {total_dets} &nbsp;|&nbsp; '
                    f'⚡ {fps_proc:.1f} fps'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            cap.release()
            writer.release()
            progress_bar.progress(1.0)

            # ── Resultado final lado a lado ───────────────────────────────
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("##### ✅ Procesamiento completado")

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown('<div class="results-box"><h3>Video original</h3>', unsafe_allow_html=True)
                st.video(tfile.name)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_v2:
                st.markdown('<div class="results-box"><h3>Video con detecciones</h3>', unsafe_allow_html=True)
                st.video(out_path)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Descarga del video procesado ──────────────────────────────
            with open(out_path, "rb") as vf:
                video_bytes = vf.read()

            st.download_button(
                label="⬇️  Descargar video con detecciones",
                data=video_bytes,
                file_name=f"deteccion_{Path(uploaded_vid.name).stem}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

            # ── Resumen estadístico ───────────────────────────────────────
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            class_counts = Counter(all_classes)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="val">{processed}</div><div class="lbl">Frames analizados</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="val">{total_dets}</div><div class="lbl">Detecciones totales</div></div>', unsafe_allow_html=True)
            with c3:
                avg_dpf = total_dets / processed if processed > 0 else 0
                st.markdown(f'<div class="metric-card"><div class="val">{avg_dpf:.1f}</div><div class="lbl">Dets / frame</div></div>', unsafe_allow_html=True)

            if class_counts:
                st.markdown('<div class="results-box" style="margin-top:1rem;"><h3>Clases detectadas en el video</h3>', unsafe_allow_html=True)
                for i, (cls_name, cnt) in enumerate(class_counts.most_common()):
                    badge = BADGE_CLASSES[i % 4]
                    st.markdown(f'<span class="badge {badge}">{cls_name} × {cnt}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Limpiar archivos temporales
            try:
                os.unlink(tfile.name)
                os.unlink(out_path)
            except Exception:
                pass

    elif uploaded_vid and not model:
        st.warning("⚠️ Carga el modelo primero desde la barra lateral.")


# ════════════════════════════════════════
#  TAB 3 — CÁMARA EN VIVO
# ════════════════════════════════════════
with tab_cam:
    st.markdown("#### 📹 Detección en tiempo real — Cámara")

    if not model:
        st.warning("⚠️ Carga el modelo primero desde la barra lateral.")
    else:
        # Instalar streamlit-webrtc si no está disponible
        WEBRTC_OK = False
        WEBRTC_ERR = ""
        try:
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
            import av
            WEBRTC_OK = True
        except Exception as _e:
            WEBRTC_ERR = str(_e)

        if not WEBRTC_OK:
            st.error("Error al cargar streamlit-webrtc: " + WEBRTC_ERR)
            st.info("Prueba: pip install streamlit-webrtc av aiortc")
        else:
            st.markdown(
                '''<div class="model-info">
                📹 La cámara corre en tiempo real en tu navegador. YOLOv8 procesa cada frame
                automáticamente y dibuja las detecciones sobre el video en vivo.
                </div>''',
                unsafe_allow_html=True,
            )

            # ── Controles ─────────────────────────────────────────────────
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                show_original_cam = st.toggle("Mostrar video original (sin detecciones)", value=False)

            # Inicializar contadores en session_state
            if "cam_frame_count" not in st.session_state:
                st.session_state["cam_frame_count"] = 0
            if "cam_total_dets" not in st.session_state:
                st.session_state["cam_total_dets"] = 0
            if "cam_class_counts" not in st.session_state:
                st.session_state["cam_class_counts"] = {}

            with col_ctrl2:
                if st.button("🔁 Resetear estadísticas", use_container_width=True):
                    st.session_state["cam_frame_count"] = 0
                    st.session_state["cam_total_dets"]  = 0
                    st.session_state["cam_class_counts"] = {}

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # ── Procesador de video WebRTC ────────────────────────────────
            # Captura el modelo y parámetros desde el scope exterior
            _model      = model
            _conf       = conf_threshold
            _iou        = iou_threshold
            _no_annot   = show_original_cam

            class MaraVideoProcessor(VideoProcessorBase):
                """
                Procesador frame a frame para streamlit-webrtc.
                Recibe cada frame de la cámara como av.VideoFrame,
                corre inferencia YOLOv8 y devuelve el frame anotado.
                """
                def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                    # Convertir frame WebRTC a numpy BGR
                    img = frame.to_ndarray(format="bgr24")

                    if _no_annot:
                        # Sin anotaciones — mostrar video limpio
                        return av.VideoFrame.from_ndarray(img, format="bgr24")

                    # Inferencia con YOLOv8
                    annotated_bgr, dets = run_inference(_model, img, _conf, _iou)

                    # Actualizar contadores en session_state (hilo seguro con Streamlit)
                    st.session_state["cam_frame_count"] = st.session_state.get("cam_frame_count", 0) + 1
                    st.session_state["cam_total_dets"]  = st.session_state.get("cam_total_dets", 0) + len(dets)
                    for d in dets:
                        cc = st.session_state.get("cam_class_counts", {})
                        cc[d["name"]] = cc.get(d["name"], 0) + 1
                        st.session_state["cam_class_counts"] = cc

                    # Convertir RGB→BGR para av (run_inference devuelve RGB)
                    out_bgr = cv2.cvtColor(annotated_bgr, cv2.COLOR_RGB2BGR)
                    return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

            # Configuración STUN/TURN para WebRTC (necesario en EC2/servidores)
            RTC_CONFIG = RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            })

            # ── Streamer principal ────────────────────────────────────────
            ctx = webrtc_streamer(
                key="mara-detector",
                video_processor_factory=MaraVideoProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # ── Estadísticas en tiempo real ────────────────────────────────
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            stat_placeholder  = st.empty()
            badge_placeholder = st.empty()

            if ctx and ctx.state.playing:
                fc  = st.session_state.get("cam_frame_count", 0)
                td  = st.session_state.get("cam_total_dets", 0)
                avg = round(td / fc, 1) if fc > 0 else 0
                stat_placeholder.markdown(
                    "<div class='results-box'><h3>Estadísticas en vivo</h3>"
                    "🎞️ Frames: <strong>" + str(fc) + "</strong> &nbsp;|&nbsp; "
                    "🔍 Detecciones: <strong>" + str(td) + "</strong> &nbsp;|&nbsp; "
                    "📊 Promedio/frame: <strong>" + str(avg) + "</strong></div>",
                    unsafe_allow_html=True,
                )
                cc = st.session_state.get("cam_class_counts", {})
                if cc:
                    sorted_cc = sorted(cc.items(), key=lambda x: -x[1])
                    badges = " ".join(
                        '<span class="badge ' + BADGE_CLASSES[i % 4] + '">' + name + ' × ' + str(cnt) + '</span>'
                        for i, (name, cnt) in enumerate(sorted_cc)
                    )
                    badge_placeholder.markdown(
                        '<div class="results-box" style="margin-top:0.5rem;"><h3>Clases detectadas</h3>' + badges + '</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="results-box" style="text-align:center; padding:2rem;">' +
                    '<div style="font-size:3rem;margin-bottom:1rem;">📹</div>' +
                    '<div style="font-size:0.93rem;color:#64748b;font-family:monospace;line-height:2;">' +
                    'Presiona <strong style="color:#00e5ff;">START</strong> para activar la cámara en tiempo real' +
                    '</div></div>',
                    unsafe_allow_html=True,
                )

            # ── Tomar foto con detección ──────────────────────────────────
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("##### 📸 Tomar foto con detección")
            st.markdown(
                '<div class="model-info">Captura un momento específico del video en vivo ' +
                'y descárgalo con las detecciones dibujadas.</div>',
                unsafe_allow_html=True,
            )

            snap = st.camera_input("Tomar foto", label_visibility="collapsed", key="cam_snap")

            if snap is not None:
                snap_bytes = np.asarray(bytearray(snap.read()), dtype=np.uint8)
                snap_bgr   = cv2.imdecode(snap_bytes, cv2.IMREAD_COLOR)

                # Inferencia sobre la foto
                t0_snap = time.time()
                snap_annotated, snap_dets = run_inference(model, snap_bgr, conf_threshold, iou_threshold)
                snap_ms = (time.time() - t0_snap) * 1000

                # Mostrar original y detección lado a lado
                col_sn1, col_sn2 = st.columns(2)
                with col_sn1:
                    st.markdown('<div class="results-box"><h3>Foto original</h3>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(snap_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_sn2:
                    st.markdown('<div class="results-box"><h3>Con detecciones</h3>', unsafe_allow_html=True)
                    st.image(snap_annotated, use_container_width=True, channels="RGB")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Métricas de la foto
                smc1, smc2, smc3 = st.columns(3)
                with smc1:
                    st.markdown(
                        '<div class="metric-card"><div class="val">' + str(len(snap_dets)) +
                        '</div><div class="lbl">Detecciones</div></div>',
                        unsafe_allow_html=True,
                    )
                with smc2:
                    avg_sc = round(sum(d["conf"] for d in snap_dets) / len(snap_dets), 2) if snap_dets else 0
                    st.markdown(
                        '<div class="metric-card"><div class="val">' + str(avg_sc) +
                        '</div><div class="lbl">Conf. media</div></div>',
                        unsafe_allow_html=True,
                    )
                with smc3:
                    st.markdown(
                        '<div class="metric-card"><div class="val">' + str(round(snap_ms)) +
                        'ms</div><div class="lbl">Inferencia</div></div>',
                        unsafe_allow_html=True,
                    )

                # Clases en la foto
                if snap_dets:
                    st.markdown(
                        '<div class="results-box" style="margin-top:0.75rem;"><h3>Clases en esta foto</h3>' +
                        detections_summary(snap_dets, model.names) + '</div>',
                        unsafe_allow_html=True,
                    )

                # Descarga de la foto con detecciones
                import io as _snap_io
                snap_pil = Image.fromarray(snap_annotated)
                snap_buf = _snap_io.BytesIO()
                snap_pil.save(snap_buf, format="PNG")
                st.download_button(
                    label="⬇️  Descargar foto con detecciones",
                    data=snap_buf.getvalue(),
                    file_name="foto_deteccion_maras.png",
                    mime="image/png",
                    use_container_width=True,
                )