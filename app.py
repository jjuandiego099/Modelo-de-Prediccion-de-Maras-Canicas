"""
Detección de Maras — Streamlit App
Proyecto: Detección de 4 tipos de maras usando YOLOv8s
Autores: Juan Diego Chaparro García, Juan José Vargas, Santiago Amado

MODO DESACOPLADO: El frontend consume la FastAPI REST en lugar de cargar
el modelo localmente. Configura API_BASE_URL para apuntar al contenedor
de inferencia (ej. http://api:8000 en docker-compose, o la IP pública de EC2).
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import io
import requests
from pathlib import Path
from PIL import Image

# ── URL de la API — configurable por variable de entorno o sidebar ─────────
import os as _os
DEFAULT_API_URL = _os.environ.get("API_BASE_URL", "http://localhost:8000")

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

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
.main-header { text-align:center; padding:2rem 0 1.5rem 0; border-bottom:1px solid var(--border); margin-bottom:2rem; }
.main-header h1 { font-family:'Syne',sans-serif; font-weight:800; font-size:2.8rem; letter-spacing:-1px; color:var(--text); margin:0; }
.main-header h1 span { color:var(--accent); }
.main-header .subtitle { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:var(--muted); margin-top:0.5rem; letter-spacing:2px; text-transform:uppercase; }
.main-header .authors { font-size:0.82rem; color:var(--muted); margin-top:0.75rem; }
.metric-card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1rem 1.25rem; text-align:center; }
.metric-card .val { font-size:2rem; font-weight:800; color:var(--accent); line-height:1; }
.metric-card .lbl { font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:1.5px; margin-top:0.25rem; }
.badge { display:inline-block; padding:0.2rem 0.65rem; border-radius:4px; font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:500; margin:0.15rem; }
.badge-0 { background:rgba(0,229,255,0.15);   color:#00e5ff; border:1px solid rgba(0,229,255,0.3); }
.badge-1 { background:rgba(255,61,87,0.15);   color:#ff3d57; border:1px solid rgba(255,61,87,0.3); }
.badge-2 { background:rgba(255,214,10,0.15);  color:#ffd60a; border:1px solid rgba(255,214,10,0.3); }
.badge-3 { background:rgba(160,110,255,0.15); color:#a06eff; border:1px solid rgba(160,110,255,0.3); }
.results-box { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1.25rem; margin-top:1rem; }
.results-box h3 { font-size:0.8rem; text-transform:uppercase; letter-spacing:2px; color:var(--muted); margin:0 0 0.75rem 0; font-weight:600; }
.model-info { background:rgba(0,229,255,0.05); border:1px solid rgba(0,229,255,0.2); border-radius:6px; padding:0.75rem 1rem; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:var(--accent); margin-bottom:1rem; }
[data-testid="stButton"] > button { background:transparent !important; border:1px solid var(--accent) !important; color:var(--accent) !important; font-family:'Syne',sans-serif !important; font-weight:600 !important; border-radius:6px !important; transition:all 0.2s !important; }
[data-testid="stButton"] > button:hover { background:var(--accent) !important; color:var(--bg) !important; }
[data-testid="stFileUploader"] { border:1px dashed var(--border) !important; border-radius:8px !important; background:var(--surface) !important; }
[data-testid="stTabs"] [role="tab"] { font-family:'Syne',sans-serif !important; font-weight:600 !important; font-size:0.88rem !important; text-transform:uppercase !important; letter-spacing:1px !important; }
[data-testid="stTabs"] [aria-selected="true"] { color:var(--accent) !important; border-bottom-color:var(--accent) !important; }
.divider { height:1px; background:linear-gradient(90deg,transparent,var(--border),transparent); margin:1.5rem 0; }
.status-pill { display:inline-flex; align-items:center; gap:0.4rem; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.78rem; font-family:'JetBrains Mono',monospace; }
.status-ok   { background:rgba(0,229,255,0.1);  color:var(--accent);  border:1px solid rgba(0,229,255,0.3); }
.status-warn { background:rgba(255,214,10,0.1); color:var(--accent3); border:1px solid rgba(255,214,10,0.3); }
.status-err  { background:rgba(255,61,87,0.1);  color:var(--accent2); border:1px solid rgba(255,61,87,0.3); }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Colores y badges ───────────────────────────────────────────────────────
CLASS_COLORS = [
    (0, 229, 255),
    (255, 61, 87),
    (255, 214, 10),
    (160, 110, 255),
]
BADGE_CLASSES = ["badge-0", "badge-1", "badge-2", "badge-3"]


# ── Helpers de API ─────────────────────────────────────────────────────────

def check_api_health(base_url: str) -> tuple[bool, dict]:
    """Llama a GET /health y retorna (ok, data)."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200:
            return True, r.json()
        return False, {"detail": r.text}
    except Exception as e:
        return False, {"detail": str(e)}


def api_predict_json(base_url: str, img_bytes: bytes, filename: str,
                     conf: float, iou: float) -> dict | None:
    """POST /predict → JSON con detecciones."""
    try:
        r = requests.post(
            f"{base_url}/predict",
            files={"file": (filename, img_bytes, "image/jpeg")},
            params={"conf": conf, "iou": iou},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except Exception as e:
        st.error(f"Error de conexión con la API: {e}")
        return None


def api_predict_image(base_url: str, img_bytes: bytes, filename: str,
                      conf: float, iou: float) -> bytes | None:
    """POST /predict/image → PNG anotado."""
    try:
        r = requests.post(
            f"{base_url}/predict/image",
            files={"file": (filename, img_bytes, "image/jpeg")},
            params={"conf": conf, "iou": iou},
            timeout=30,
        )
        if r.status_code == 200:
            return r.content
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except Exception as e:
        st.error(f"Error de conexión con la API: {e}")
        return None


def api_predict_video(base_url: str, video_bytes: bytes, filename: str,
                      conf: float, iou: float,
                      skip: int = 1) -> bytes | None:
    """POST /predict/video → MP4 anotado."""
    try:
        r = requests.post(
            f"{base_url}/predict/video",
            files={"file": (filename, video_bytes, "video/mp4")},
            params={"conf": conf, "iou": iou, "skip_frames": skip},
            timeout=300,  # videos pueden tardar
        )
        if r.status_code == 200:
            return r.content
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except Exception as e:
        st.error(f"Error de conexión con la API: {e}")
        return None


def draw_detections_local(img_bgr: np.ndarray, detections: list) -> np.ndarray:
    """
    Dibuja bboxes sobre imagen BGR usando las detecciones JSON de /predict.
    Retorna imagen RGB para Streamlit.
    """
    annotated = img_bgr.copy()
    for det in detections:
        cls_id = det["class_id"]
        color  = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']}  {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (10, 12, 15), 1, cv2.LINE_AA)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def detections_summary(detections: list) -> str:
    """Genera HTML con badges de resumen de detecciones (formato JSON de API)."""
    if not detections:
        return '<span style="color:#64748b;font-size:0.85rem;">Sin detecciones</span>'
    from collections import Counter
    counts = Counter(d["class_name"] for d in detections)
    parts  = []
    for i, (name, count) in enumerate(counts.items()):
        cls_ids   = [d["class_id"] for d in detections if d["class_name"] == name]
        badge_cls = BADGE_CLASSES[cls_ids[0] % 4] if cls_ids else "badge-0"
        parts.append(f'<span class="badge {badge_cls}">{name} × {count}</span>')
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════
#  LAYOUT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <h1>🔮 Detección de <span>Maras</span></h1>
  <p class="subtitle">Sistema de Visión por Computador · YOLOv8s</p>
  <p class="authors">Juan Diego Chaparro García &nbsp;·&nbsp; Juan José Vargas &nbsp;·&nbsp; Santiago Amado</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 ¿Qué es este proyecto y cómo funciona?", expanded=False):
    col_desc, col_clases = st.columns([3, 2])
    with col_desc:
        st.markdown("""
<div style="font-family:sans-serif; line-height:1.8;">
<h4 style="color:#00e5ff;margin-top:0;font-size:1rem;text-transform:uppercase;letter-spacing:2px;">¿Qué son las maras?</h4>
<p style="color:#e8edf2;font-size:0.93rem;">
En la región de Santander (Colombia), las <strong style="color:#00e5ff;">maras</strong> son las <strong>canicas</strong>,
pequeñas esferas de vidrio de colores. Este proyecto aplica <strong>Inteligencia Artificial</strong> para detectar y
clasificar automáticamente 4 tipos según su color usando YOLOv8s.
</p>
<h4 style="color:#00e5ff;font-size:1rem;text-transform:uppercase;letter-spacing:2px;">Arquitectura</h4>
<p style="color:#e8edf2;font-size:0.93rem;">
El frontend (Streamlit) está desacoplado del modelo de inferencia (FastAPI + YOLOv8s).
Ambos corren como contenedores Docker independientes en EC2 y se comunican por HTTP.
</p>
</div>""", unsafe_allow_html=True)
    with col_clases:
        st.markdown("""
<div style="font-family:sans-serif;">
<h4 style="color:#00e5ff;margin-top:0;font-size:1rem;text-transform:uppercase;letter-spacing:2px;">Clases detectadas</h4>
<div style="display:flex;flex-direction:column;gap:0.75rem;margin-top:1rem;">
  <div style="background:#111418;border:1px solid #1e2530;border-left:4px solid #00c853;border-radius:6px;padding:0.75rem 1rem;">
    <span style="font-size:1.2rem;">🟢</span> <strong style="color:#00c853;">Mara Verde</strong>
    <div style="color:#64748b;font-size:0.78rem;font-family:monospace;">green marble · 124 muestras</div>
  </div>
  <div style="background:#111418;border:1px solid #1e2530;border-left:4px solid #2979ff;border-radius:6px;padding:0.75rem 1rem;">
    <span style="font-size:1.2rem;">🔵</span> <strong style="color:#2979ff;">Mara Azul</strong>
    <div style="color:#64748b;font-size:0.78rem;font-family:monospace;">blue marble · 147 muestras</div>
  </div>
  <div style="background:#111418;border:1px solid #1e2530;border-left:4px solid #e0e0e0;border-radius:6px;padding:0.75rem 1rem;">
    <span style="font-size:1.2rem;">⚪</span> <strong style="color:#e0e0e0;">Mara Blanca</strong>
    <div style="color:#64748b;font-size:0.78rem;font-family:monospace;">white marble · 144 muestras</div>
  </div>
  <div style="background:#111418;border:1px solid #1e2530;border-left:4px solid #ffd60a;border-radius:6px;padding:0.75rem 1rem;">
    <span style="font-size:1.2rem;">⚫</span> <strong style="color:#ffd60a;">Mara Negra</strong>
    <div style="color:#64748b;font-size:0.78rem;font-family:monospace;">black marble · 116 muestras</div>
  </div>
</div>
</div>""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    # URL de la API
    st.markdown("**🌐 URL de la API de inferencia**")
    api_base_url = st.text_input(
        "api_url",
        value=DEFAULT_API_URL,
        label_visibility="collapsed",
        placeholder="http://localhost:8000",
    )
    api_base_url = api_base_url.rstrip("/")

    # Verificar conexión con la API
    if st.button("🔌 Verificar conexión", use_container_width=True):
        with st.spinner("Conectando..."):
            ok, info = check_api_health(api_base_url)
        if ok:
            st.markdown(
                f'<div class="status-ok">✓ API lista &nbsp;|&nbsp; {info.get("num_classes","?")} clases</div>',
                unsafe_allow_html=True,
            )
            st.session_state["api_ok"]    = True
            st.session_state["api_names"] = info.get("classes", {})
        else:
            st.markdown(
                f'<div class="status-err">❌ Sin conexión: {info.get("detail","")}</div>',
                unsafe_allow_html=True,
            )
            st.session_state["api_ok"] = False

    # Estado actual guardado en session
    api_ok = st.session_state.get("api_ok", None)
    if api_ok is True:
        st.markdown('<div class="model-info">📡 Conectado a la API de inferencia</div>', unsafe_allow_html=True)
    elif api_ok is False:
        st.markdown('<div class="status-warn">⚠️ API no disponible — verifica la URL y que el contenedor esté corriendo.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-info">ℹ️ Presiona "Verificar conexión" para comprobar la API</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("**🎯 Umbral de confianza**")
    conf_threshold = st.slider("conf", 0.1, 1.0, 0.60, 0.05, label_visibility="collapsed")
    with st.expander("ℹ️ ¿Qué es?"):
        st.markdown(
            '<div style="font-size:0.82rem;color:#94a3b8;line-height:1.7;">'
            'Porcentaje mínimo de certeza para reportar una detección.<br><br>'
            '<b style="color:#00e5ff;">Valor alto (0.7–0.9):</b> Menos detecciones, más precisas.<br>'
            '<b style="color:#ffd60a;">Valor bajo (0.2–0.4):</b> Más detecciones, posibles falsos positivos.<br><br>'
            '<i>Recomendado: 0.50–0.65</i></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("**📐 Umbral IoU (NMS)**")
    iou_threshold = st.slider("iou", 0.1, 1.0, 0.45, 0.05, label_visibility="collapsed")
    with st.expander("ℹ️ ¿Qué es?"):
        st.markdown(
            '<div style="font-size:0.82rem;color:#94a3b8;line-height:1.7;">'
            'Controla NMS para eliminar cajas duplicadas.<br><br>'
            '<b style="color:#00e5ff;">Valor alto (0.6–0.9):</b> Permite más cajas superpuestas.<br>'
            '<b style="color:#ffd60a;">Valor bajo (0.2–0.4):</b> Elimina más duplicados.<br><br>'
            '<i>Recomendado: 0.40–0.50</i></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.72rem;color:#64748b;text-align:center;font-family:\'JetBrains Mono\',monospace;">'
        'Detección · 4 clases de maras<br>Desplegado en AWS EC2</p>',
        unsafe_allow_html=True,
    )

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

    if uploaded_img:
        img_bytes = uploaded_img.read()
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col_orig, col_pred = st.columns(2)
        with col_orig:
            st.markdown('<div class="results-box"><h3>Original</h3>', unsafe_allow_html=True)
            st.image(img_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("Enviando imagen a la API..."):
            t0       = time.time()
            result   = api_predict_json(api_base_url, img_bytes, uploaded_img.name,
                                        conf_threshold, iou_threshold)
            elapsed  = time.time() - t0

        if result:
            detections   = result["detections"]
            annotated_rgb = draw_detections_local(img_bgr, detections)

            with col_pred:
                st.markdown('<div class="results-box"><h3>Detecciones</h3>', unsafe_allow_html=True)
                st.image(annotated_rgb, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Métricas
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            h, w = img_bgr.shape[:2]
            avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0
            with m1:
                st.markdown(f'<div class="metric-card"><div class="val">{len(detections)}</div><div class="lbl">Detecciones</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="val">{avg_conf:.2f}</div><div class="lbl">Conf. media</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="val">{result["inference_ms"]:.0f}ms</div><div class="lbl">Inferencia (API)</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="val">{w}×{h}</div><div class="lbl">Resolución</div></div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="results-box" style="margin-top:1rem;"><h3>Clases encontradas</h3>{detections_summary(detections)}</div>',
                unsafe_allow_html=True,
            )

            # Tabla de detecciones
            if detections:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                with st.expander("📋 Tabla de detecciones"):
                    import pandas as pd
                    rows = [{
                        "#": i + 1,
                        "Clase": d["class_name"],
                        "Confianza": f"{d['confidence']:.4f}",
                        "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                        "Ancho": d["width"], "Alto": d["height"],
                    } for i, d in enumerate(detections)]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Descarga — usa /predict/image para imagen anotada por la API
            with st.spinner("Generando imagen anotada para descarga..."):
                annotated_png = api_predict_image(api_base_url, img_bytes, uploaded_img.name,
                                                   conf_threshold, iou_threshold)
            if annotated_png:
                st.download_button(
                    label="⬇️  Descargar imagen con detecciones",
                    data=annotated_png,
                    file_name=f"deteccion_{uploaded_img.name}",
                    mime="image/png",
                )


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

    if uploaded_vid:
        video_bytes = uploaded_vid.read()

        # Guardar temporalmente para leer metadatos con OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_vid.name).suffix)
        tfile.write(video_bytes)
        tfile.flush()
        tfile.close()

        cap          = cv2.VideoCapture(tfile.name)
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

        st.markdown('<div class="results-box"><h3>Video original</h3>', unsafe_allow_html=True)
        st.video(tfile.name)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        skip_frames = st.slider(
            "Procesar 1 de cada N frames",
            1, 10, 1, 1,
            help="1 = todos los frames (más lento, mejor resultado). La API aplica el skip internamente.",
        )

        if st.button("▶️  Procesar video con detección", use_container_width=True):
            with st.spinner("Enviando video a la API para inferencia... (puede tardar varios minutos)"):
                t_start     = time.time()
                result_mp4  = api_predict_video(
                    api_base_url, video_bytes, uploaded_vid.name,
                    conf_threshold, iou_threshold, skip_frames,
                )
                elapsed_vid = time.time() - t_start

            if result_mp4:
                # Guardar resultado para reproducir
                out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out_file.write(result_mp4)
                out_file.flush()
                out_file.close()

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("##### ✅ Procesamiento completado")

                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.markdown('<div class="results-box"><h3>Video original</h3>', unsafe_allow_html=True)
                    st.video(tfile.name)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_v2:
                    st.markdown('<div class="results-box"><h3>Video con detecciones</h3>', unsafe_allow_html=True)
                    st.video(out_file.name)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(
                    f'<div class="metric-card" style="margin-top:1rem;">'
                    f'<div class="val">{elapsed_vid:.1f}s</div>'
                    f'<div class="lbl">Tiempo total de procesamiento</div></div>',
                    unsafe_allow_html=True,
                )

                st.download_button(
                    label="⬇️  Descargar video con detecciones",
                    data=result_mp4,
                    file_name=f"deteccion_{Path(uploaded_vid.name).stem}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

                try:
                    os.unlink(out_file.name)
                except Exception:
                    pass

        try:
            os.unlink(tfile.name)
        except Exception:
            pass


# ════════════════════════════════════════
#  TAB 3 — CÁMARA EN VIVO
# ════════════════════════════════════════
with tab_cam:
    st.markdown("#### 📹 Detección en tiempo real — Cámara")
    st.markdown(
        '<div class="model-info">ℹ️ En modo cámara cada foto capturada se envía a la API. '
        'Para video en tiempo real real se recomienda usar el cliente directamente contra la API.</div>',
        unsafe_allow_html=True,
    )

    # Inicializar contadores en session_state
    for key, default in [("cam_frame_count", 0), ("cam_total_dets", 0), ("cam_class_counts", {})]:
        if key not in st.session_state:
            st.session_state[key] = default

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        show_original_cam = st.toggle("Mostrar foto sin detecciones", value=False)
    with col_ctrl2:
        if st.button("🔁 Resetear estadísticas", use_container_width=True):
            st.session_state["cam_frame_count"] = 0
            st.session_state["cam_total_dets"]  = 0
            st.session_state["cam_class_counts"] = {}

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── WebRTC (stream en vivo) ────────────────────────────────────────────
    WEBRTC_OK  = False
    WEBRTC_ERR = ""
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av
        WEBRTC_OK = True
    except Exception as _e:
        WEBRTC_ERR = str(_e)

    if not WEBRTC_OK:
        st.warning("streamlit-webrtc no disponible: " + WEBRTC_ERR)
        st.info("Instala con: pip install streamlit-webrtc av aiortc")
    else:
        _api_url = api_base_url
        _conf    = conf_threshold
        _iou     = iou_threshold
        _no_ann  = show_original_cam

        class MaraVideoProcessor(VideoProcessorBase):
            """
            Procesador frame a frame para streamlit-webrtc.
            Cada frame se envía a POST /predict/image de la API y se devuelve anotado.
            """
            def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                img_bgr = frame.to_ndarray(format="bgr24")

                if _no_ann:
                    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

                # Codificar frame como JPEG y enviar a la API
                _, buf = cv2.imencode(".jpg", img_bgr)
                jpg_bytes = buf.tobytes()

                try:
                    r = requests.post(
                        f"{_api_url}/predict/image",
                        files={"file": ("frame.jpg", jpg_bytes, "image/jpeg")},
                        params={"conf": _conf, "iou": _iou},
                        timeout=5,
                    )
                    if r.status_code == 200:
                        # Decodificar PNG anotado que devuelve la API
                        nparr      = np.frombuffer(r.content, np.uint8)
                        annotated  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        num_dets   = int(r.headers.get("X-Detections", 0))

                        # Actualizar contadores
                        st.session_state["cam_frame_count"] = st.session_state.get("cam_frame_count", 0) + 1
                        st.session_state["cam_total_dets"]  = st.session_state.get("cam_total_dets", 0) + num_dets

                        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                except Exception:
                    pass  # Si la API falla, devolver frame sin anotar

                return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

        RTC_CONFIG = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        })

        ctx = webrtc_streamer(
            key="mara-detector",
            video_processor_factory=MaraVideoProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

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
        else:
            st.markdown(
                '<div class="results-box" style="text-align:center;padding:2rem;">'
                '<div style="font-size:3rem;margin-bottom:1rem;">📹</div>'
                '<div style="font-size:0.93rem;color:#64748b;font-family:monospace;line-height:2;">'
                'Presiona <strong style="color:#00e5ff;">START</strong> para activar la cámara. '
                'Cada frame se envía a la API de inferencia.</div></div>',
                unsafe_allow_html=True,
            )

    # ── Foto única con detección (funciona sin WebRTC) ─────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("##### 📸 Tomar foto con detección")
    st.markdown(
        '<div class="model-info">Captura un momento y envíalo a la API para detección.</div>',
        unsafe_allow_html=True,
    )

    snap = st.camera_input("Tomar foto", label_visibility="collapsed", key="cam_snap")

    if snap is not None:
        snap_bytes = snap.read()
        nparr      = np.frombuffer(snap_bytes, np.uint8)
        snap_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with st.spinner("Detectando..."):
            t0_snap    = time.time()
            snap_result = api_predict_json(api_base_url, snap_bytes, "snap.jpg",
                                           conf_threshold, iou_threshold)
            snap_ms    = (time.time() - t0_snap) * 1000

        if snap_result:
            snap_dets      = snap_result["detections"]
            snap_annotated = draw_detections_local(snap_bgr, snap_dets)

            col_sn1, col_sn2 = st.columns(2)
            with col_sn1:
                st.markdown('<div class="results-box"><h3>Foto original</h3>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(snap_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_sn2:
                st.markdown('<div class="results-box"><h3>Con detecciones</h3>', unsafe_allow_html=True)
                st.image(snap_annotated, use_container_width=True, channels="RGB")
                st.markdown('</div>', unsafe_allow_html=True)

            smc1, smc2, smc3 = st.columns(3)
            avg_sc = round(sum(d["confidence"] for d in snap_dets) / len(snap_dets), 2) if snap_dets else 0
            with smc1:
                st.markdown(f'<div class="metric-card"><div class="val">{len(snap_dets)}</div><div class="lbl">Detecciones</div></div>', unsafe_allow_html=True)
            with smc2:
                st.markdown(f'<div class="metric-card"><div class="val">{avg_sc}</div><div class="lbl">Conf. media</div></div>', unsafe_allow_html=True)
            with smc3:
                st.markdown(f'<div class="metric-card"><div class="val">{snap_result["inference_ms"]:.0f}ms</div><div class="lbl">Inferencia (API)</div></div>', unsafe_allow_html=True)

            if snap_dets:
                st.markdown(
                    '<div class="results-box" style="margin-top:0.75rem;"><h3>Clases en esta foto</h3>' +
                    detections_summary(snap_dets) + '</div>',
                    unsafe_allow_html=True,
                )

            snap_pil = Image.fromarray(snap_annotated)
            snap_buf = io.BytesIO()
            snap_pil.save(snap_buf, format="PNG")
            st.download_button(
                label="⬇️  Descargar foto con detecciones",
                data=snap_buf.getvalue(),
                file_name="foto_deteccion_maras.png",
                mime="image/png",
                use_container_width=True,
            )