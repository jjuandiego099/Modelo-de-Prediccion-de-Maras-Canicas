"""
Detección de Maras — Streamlit App
Proyecto: Detección de 4 tipos de maras usando YOLOv8s
Autores: Juan Diego Chaparro García, Juan José Vargas, Santiago Amado

MODO DESACOPLADO: El frontend consume la FastAPI REST en lugar de cargar
el modelo localmente. Configura API_BASE_URL para apuntar al contenedor
de inferencia (ej. http://api:8000 en docker-compose, o la IP pública de EC2).
"""

import streamlit as st
import base64
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
DEFAULT_API_URL = "http://api:8000"  # cambia esto después
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
.stat-row { display:flex; align-items:center; justify-content:space-between; padding:0.65rem 1rem; border-bottom:1px solid var(--border); font-size:0.88rem; }
.stat-row:last-child { border-bottom:none; }
.stat-row .idx { color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:0.75rem; width:2rem; }
.stat-row .fecha { color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:0.75rem; }
.stat-row .fuente-badge { padding:0.15rem 0.5rem; border-radius:4px; font-size:0.72rem; font-family:'JetBrains Mono',monospace; }
.fuente-imagen  { background:rgba(0,229,255,0.1);   color:#00e5ff;  border:1px solid rgba(0,229,255,0.2); }
.fuente-video   { background:rgba(160,110,255,0.1); color:#a06eff;  border:1px solid rgba(160,110,255,0.2); }
.fuente-camara  { background:rgba(255,214,10,0.1);  color:#ffd60a;  border:1px solid rgba(255,214,10,0.2); }
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
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200:
            return True, r.json()
        return False, {"detail": r.text}
    except Exception as e:
        return False, {"detail": str(e)}


def api_predict_json(base_url, img_bytes, filename, conf, iou):
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


def api_predict_image(base_url, img_bytes, filename, conf, iou):
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


def api_predict_video(base_url, video_bytes, filename, conf, iou, skip=1):
    try:
        r = requests.post(
            f"{base_url}/predict/video",
            files={"file": (filename, video_bytes, "video/mp4")},
            params={"conf": conf, "iou": iou, "skip_frames": skip},
            timeout=300,
        )
        if r.status_code == 200:
            return r.content
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except Exception as e:
        st.error(f"Error de conexión con la API: {e}")
        return None


def draw_detections_local(img_bgr, detections):
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


def detections_summary(detections):
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


# ── Helpers de Base de Datos ───────────────────────────────────────────────

def get_db_connection():
    """Retorna una conexión a PostgreSQL usando variables de entorno."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="postgres",
            port=_os.getenv("DB_PORT", "5432"),
            dbname=_os.getenv("DB_NAME", "detecciones"),
            user=_os.getenv("DB_USER", "admin"),
            password=_os.getenv("DB_PASS", "canicas123"),
            connect_timeout=5,
        )
        return conn
    except Exception:
        return None


def guardar_deteccion(fuente: str, detections: list, inference_ms: float):
    from collections import Counter
    counts = Counter(d["class_name"].lower() for d in detections)
    verde  = counts.get("green marble", 0)
    azul   = counts.get("blue marble", 0)
    blanca = counts.get("white marble", 0)
    negra  = counts.get("black marble", 0)
    total  = len(detections)
    conf_avg = round(float(np.mean([d["confidence"] for d in detections])), 4) if detections else 0.0

    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO detecciones (fuente, verde, azul, blanca, negra, total, confianza_avg, inferencia_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (fuente, verde, azul, blanca, negra, total, conf_avg, inference_ms))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        conn.close()
        return False


def obtener_detecciones(limit: int = 10, offset: int = 0):
    """Retorna las últimas detecciones con paginación."""
    conn = get_db_connection()
    if not conn:
        return [], 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM detecciones")
        total = cur.fetchone()[0]
        cur.execute("""
            SELECT id, fecha, fuente, verde, azul, blanca, negra, total, confianza_avg, inferencia_ms
            FROM detecciones
            ORDER BY fecha DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows, total
    except Exception:
        conn.close()
        return [], 0


def obtener_totales_por_clase():
    """Retorna suma total de cada clase para el gráfico comparativo."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                SUM(verde)  AS verde,
                SUM(azul)   AS azul,
                SUM(blanca) AS blanca,
                SUM(negra)  AS negra
            FROM detecciones
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        return {"Verde": row[0] or 0, "Azul": row[1] or 0, "Blanca": row[2] or 0, "Negra": row[3] or 0}
    except Exception:
        conn.close()
        return None


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

    ok, info = check_api_health(DEFAULT_API_URL)
    if ok:
        st.markdown('<div class="status-ok">✓ API conectada</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-err">❌ API no disponible</div>', unsafe_allow_html=True)
    api_base_url = DEFAULT_API_URL.rstrip("/")

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
tab_img, tab_vid, tab_cam, tab_stats = st.tabs([
    "📷  Imagen", "🎬  Video", "📹  Cámara en vivo", "📊  Estadísticas"
])


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
            t0      = time.time()
            result  = api_predict_json(api_base_url, img_bytes, uploaded_img.name,
                                       conf_threshold, iou_threshold)
            elapsed = time.time() - t0

        if result:
            detections    = result["detections"]
            annotated_rgb = draw_detections_local(img_bgr, detections)

            with col_pred:
                st.markdown('<div class="results-box"><h3>Detecciones</h3>', unsafe_allow_html=True)
                st.image(annotated_rgb, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Guardar en BD
            guardar_deteccion("imagen", detections, result["inference_ms"])

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
        clave_pred  = f"pred_{uploaded_vid.name}_{len(video_bytes)}"
        clave_meta  = f"meta_{uploaded_vid.name}_{len(video_bytes)}"

        if f"orig_{clave_pred}" not in st.session_state:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_vid.name).suffix)
            tfile.write(video_bytes)
            tfile.flush()
            tfile.close()
            st.session_state[f"orig_{clave_pred}"] = tfile.name

        orig_path = st.session_state[f"orig_{clave_pred}"]
        col_orig, col_pred = st.columns(2, gap="large")

        with col_orig:
            st.markdown("#### 🎬 Video original")
            with open(orig_path, "rb") as f:
                orig_b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f'<video controls style="width:100%;border-radius:6px;">'
                f'<source src="data:video/mp4;base64,{orig_b64}" type="video/mp4">'
                f'</video>',
                unsafe_allow_html=True,
            )

        with col_pred:
            st.markdown("#### 🔍 Video con detecciones")
            if clave_pred not in st.session_state:
                with st.spinner("Procesando..."):
                    t_start    = time.time()
                    result_mp4 = api_predict_video(
                        api_base_url, video_bytes, uploaded_vid.name,
                        conf_threshold, iou_threshold, 1,
                    )
                    elapsed = time.time() - t_start

                if result_mp4:
                    st.session_state[clave_pred] = base64.b64encode(result_mp4).decode()
                    st.session_state[clave_meta] = {"tiempo": elapsed}
                else:
                    st.error("Error al procesar. Verifica que la API esté corriendo.")

            if clave_pred in st.session_state:
                vid_b64 = st.session_state[clave_pred]
                meta    = st.session_state[clave_meta]
                st.markdown(
                    f'<video controls style="width:100%;border-radius:6px;">'
                    f'<source src="data:video/mp4;base64,{vid_b64}" type="video/mp4">'
                    f'</video>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="model-info">✅ Procesado en {meta["tiempo"]:.1f}s</div>', unsafe_allow_html=True)
                st.download_button(
                    label="⬇️ Descargar video con detecciones",
                    data=base64.b64decode(vid_b64),
                    file_name=f"deteccion_{Path(uploaded_vid.name).stem}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )


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
            def recv(self, frame):
                img_bgr = frame.to_ndarray(format="bgr24")
                if _no_ann:
                    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")
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
                        nparr     = np.frombuffer(r.content, np.uint8)
                        annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        num_dets  = int(r.headers.get("X-Detections", 0))
                        st.session_state["cam_frame_count"] = st.session_state.get("cam_frame_count", 0) + 1
                        st.session_state["cam_total_dets"]  = st.session_state.get("cam_total_dets", 0) + num_dets
                        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                except Exception:
                    pass
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
            t0_snap     = time.time()
            snap_result = api_predict_json(api_base_url, snap_bytes, "snap.jpg",
                                           conf_threshold, iou_threshold)
            snap_ms     = (time.time() - t0_snap) * 1000

        if snap_result:
            snap_dets      = snap_result["detections"]
            snap_annotated = draw_detections_local(snap_bgr, snap_dets)

            # Guardar en BD
            guardar_deteccion("camara", snap_dets, snap_result["inference_ms"])

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


# ════════════════════════════════════════
#  TAB 4 — ESTADÍSTICAS
# ════════════════════════════════════════
with tab_stats:
    import pandas as pd

    st.markdown("#### 📊 Estadísticas de detecciones")

    # ── Verificar conexión a BD ────────────────────────────────────────────
    conn_test = get_db_connection()
    if not conn_test:
        st.markdown(
            '<div class="results-box" style="text-align:center;padding:2rem;">'
            '<div style="font-size:2.5rem;margin-bottom:1rem;">🗄️</div>'
            '<div style="color:#64748b;font-family:monospace;font-size:0.9rem;">'
            'Base de datos no disponible.<br>Verifica que el contenedor PostgreSQL esté corriendo.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        conn_test.close()

        # ── Botón de refresco ──────────────────────────────────────────────
        col_ref, col_empty = st.columns([1, 4])
        with col_ref:
            if st.button("🔄 Actualizar", use_container_width=True, type="primary"):
                st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Gráfico comparativo entre las 4 clases ─────────────────────────
        totales = obtener_totales_por_clase()

        if totales and sum(totales.values()) > 0:
            st.markdown("##### Distribución total por clase")

            col_chart, col_summary = st.columns([3, 2])

            with col_chart:
                df_chart = pd.DataFrame({
                    "Clase": list(totales.keys()),
                    "Total": list(totales.values()),
                    "Color": ["#00c853", "#2979ff", "#e0e0e0", "#ffd60a"],
                })
                # Gráfico de barras con Streamlit nativo estilizado
                st.bar_chart(
                    df_chart.set_index("Clase")["Total"],
                    color=["#00e5ff"],
                    use_container_width=True,
                    height=280,
                )

            with col_summary:
                total_general = sum(totales.values())
               
                st.markdown('<h3>Resumen</h3>', unsafe_allow_html=True)

                clases_info = [
                    ("🟢", "Verde",  totales["Verde"],  "#00c853"),
                    ("🔵", "Azul",   totales["Azul"],   "#2979ff"),
                    ("⚪", "Blanca", totales["Blanca"], "#e0e0e0"),
                    ("⚫", "Negra",  totales["Negra"],  "#ffd60a"),
                ]
                for emoji, nombre, cantidad, color in clases_info:
                    pct = round(cantidad / total_general * 100, 1) if total_general > 0 else 0
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:0.5rem 0;border-bottom:1px solid #1e2530;">'
                        f'<span>{emoji} <strong style="color:{color};">{nombre}</strong></span>'
                        f'<span style="font-family:monospace;color:#e8edf2;">{cantidad} '
                        f'<span style="color:#64748b;font-size:0.78rem;">({pct}%)</span></span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div style="padding:0.6rem 0;text-align:right;">'
                    f'<span style="color:#64748b;font-size:0.8rem;">Total: </span>'
                    f'<strong style="color:#00e5ff;font-size:1.1rem;">{total_general}</strong></div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="model-info">📭 Aún no hay detecciones registradas. '
                'Usa los tabs de Imagen, Video o Cámara para empezar.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Tabla paginada de últimos resultados ───────────────────────────
        st.markdown("##### Historial de detecciones")

        PAGE_SIZE = 10

        if "stats_page" not in st.session_state:
            st.session_state["stats_page"] = 0

        rows, total_rows = obtener_detecciones(
            limit=PAGE_SIZE,
            offset=st.session_state["stats_page"] * PAGE_SIZE,
        )
        total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

        if rows:
            # Tabla estilizada
            st.markdown('<div class="results-box" style="padding:0;">', unsafe_allow_html=True)

            # Encabezado
            st.markdown(
                '<div style="display:grid;grid-template-columns:2rem 5.5rem 4.5rem 1fr 1fr 1fr 1fr 3rem 3.5rem 4rem;'
                'gap:0.5rem;padding:0.6rem 1rem;border-bottom:1px solid #1e2530;'
                'font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:600;">'
                '<span>#</span><span>Fecha</span><span>Fuente</span>'
                '<span>🟢 Verde</span><span>🔵 Azul</span><span>⚪ Blanca</span><span>⚫ Negra</span>'
                '<span>Total</span><span>Conf.</span><span>Ms</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            for row in rows:
                rid, fecha, fuente, verde, azul, blanca, negra, total, conf, ms = row
                fuente_class = f"fuente-{fuente}" if fuente in ["imagen", "video", "camara"] else "fuente-imagen"
                fecha_str = fecha.strftime("%m/%d %H:%M") if fecha else "--"
                st.markdown(
                    f'<div style="display:grid;grid-template-columns:2rem 5.5rem 4.5rem 1fr 1fr 1fr 1fr 3rem 3.5rem 4rem;'
                    f'gap:0.5rem;padding:0.6rem 1rem;border-bottom:1px solid #1e2530;font-size:0.83rem;align-items:center;">'
                    f'<span style="color:#64748b;font-family:monospace;font-size:0.72rem;">{rid}</span>'
                    f'<span style="color:#64748b;font-family:monospace;font-size:0.72rem;">{fecha_str}</span>'
                    f'<span><span class="fuente-badge {fuente_class}">{fuente}</span></span>'
                    f'<span style="color:#00c853;font-family:monospace;">{verde}</span>'
                    f'<span style="color:#2979ff;font-family:monospace;">{azul}</span>'
                    f'<span style="color:#e0e0e0;font-family:monospace;">{blanca}</span>'
                    f'<span style="color:#ffd60a;font-family:monospace;">{negra}</span>'
                    f'<span style="color:#00e5ff;font-weight:700;font-family:monospace;">{total}</span>'
                    f'<span style="color:#64748b;font-family:monospace;font-size:0.78rem;">{conf:.2f}</span>'
                    f'<span style="color:#64748b;font-family:monospace;font-size:0.78rem;">{ms:.0f}ms</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Paginación ─────────────────────────────────────────────────
            st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)
            col_prev, col_info, col_next = st.columns([1, 3, 1])

            current_page = st.session_state["stats_page"]

            with col_prev:
                if st.button("← Anterior", use_container_width=True,
                             disabled=current_page == 0):
                    st.session_state["stats_page"] -= 1
                    st.rerun()

            with col_info:
                inicio = current_page * PAGE_SIZE + 1
                fin    = min(inicio + PAGE_SIZE - 1, total_rows)
                st.markdown(
                    f'<div style="text-align:center;color:#64748b;font-family:monospace;font-size:0.82rem;padding-top:0.5rem;">'
                    f'Mostrando {inicio}–{fin} de {total_rows} registros &nbsp;·&nbsp; '
                    f'Página {current_page + 1} de {total_pages}</div>',
                    unsafe_allow_html=True,
                )

            with col_next:
                if st.button("Siguiente →", use_container_width=True,
                             disabled=current_page >= total_pages - 1):
                    st.session_state["stats_page"] += 1
                    st.rerun()
        else:
            st.markdown(
                '<div class="model-info">📭 No hay registros en esta página.</div>',
                unsafe_allow_html=True,
            )