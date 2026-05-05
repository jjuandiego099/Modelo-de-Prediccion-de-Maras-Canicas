# ── Imagen base ───────────────────────────────────────────────────────────
# Python 3.11 slim — liviana, sin GUI, ideal para EC2 sin GPU
FROM python:3.12-slim

# ── Variables de entorno ───────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT_API=8000 \
    PORT_APP=8501

# ── Dependencias del sistema ───────────────────────────────────────────────
# libgl1 y libglib2.0 son necesarias para OpenCV en Linux sin pantalla
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Directorio de trabajo ──────────────────────────────────────────────────
WORKDIR /app

# ── Instalar dependencias Python ───────────────────────────────────────────
# Primero solo requirements para aprovechar cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copiar código fuente ───────────────────────────────────────────────────
COPY . .

# ── Exponer puertos ────────────────────────────────────────────────────────
# 8000 → FastAPI
# 8501 → Streamlit
EXPOSE 8000 8501

# ── Script de arranque ─────────────────────────────────────────────────────
# Arranca FastAPI y Streamlit en paralelo
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]