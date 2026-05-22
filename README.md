# 🔮 Detección de Maras (Canicas) — YOLOv8s

> Sistema de visión por computador para detección y clasificación automática de 4 tipos de canicas usando YOLOv8s, desplegado en AWS EC2 con Docker, HTTPS y base de datos PostgreSQL.

**Autores:** Juan Diego Chaparro García · Juan José Vargas · Santiago Amado  
**Universidad:** Universidad Autonoma de Bucaramanga · **Curso:**  Inteligencia Artificial y Ciencia de Datos  
**Año:** 2026

---

## 📖 ¿Qué son las Maras?

En la región de **Santander, Colombia**, las *maras* es el nombre popular para las **canicas** — pequeñas esferas de vidrio de colores usadas en juegos tradicionales. Este proyecto aplica inteligencia artificial para detectar y clasificar automáticamente 4 tipos de maras según su color a partir de imágenes, videos y cámara en tiempo real.

---

## 🗂️ Estructura del Proyecto

```
Modelo-de-Prediccion-de-Maras-Canicas/
│
├── app.py                  # Frontend Streamlit — interfaz web completa
├── api.py                  # Backend FastAPI — servidor de inferencia YOLOv8s
├── modelo_maras.ipynb      # Cuaderno Jupyter — entrenamiento y evaluación del modelo
├── best.pt                 # Pesos del modelo YOLOv8s entrenado
├── best.onnx               # Pesos del modelo YOLOv8s entrenado formato onnx
├── requirements.txt        # Dependencias Python
├── Dockerfile              # Imagen Docker unificada (API + Streamlit + Postgres)
├── docker-compose.yml      # Orquestación de todos los servicios
├── dockerignore            # Archivos ignorados por docker 
├── MD                      # Carpeta relacionada con archivos .MD  
  ├── API.MD                # Explicacion del API 
  ├── DEPLOYMENT.MD         # Explicacion del deployment
├── Dockerfile              # Imagen Docker unificada (API + Streamlit + Postgres)
├── Deployment.md           # Instrucciones de despliegue
└── README.md               # Este archivo

```

---


## 🧠 Modelo — `modelo_maras.ipynb`
El cuaderno de Jupyter documenta todo el pipeline de entrenamiento del modelo de detección

### Contenido del cuaderno
- **Exploración del dataset** — análisis de las 4 clases, distribución de muestras y visualización de imágenes etiquetadas
- **Preprocesamiento** — redimensionado, augmentación de datos (flip, rotación, brillo) y división train/val/test
- **Entrenamiento con YOLOv8s** — configuración de hiperparámetros, número de épocas, batch size y optimizador
- **Evaluación** — métricas mAP@0.5, precisión, recall y curvas F1 por clase
- **Exportación** — guardado de los pesos `best.pt` para producción

## 🧠 Dataset de Maras o Canicas 
- Link [Dataset](https://app.roboflow.com/juan-jose-vargas-correa-s-workspace/deteccion-maras-yb0z3/4)

### Clases detectadas

| ID | Clase | Color | Muestras |
|----|-------|-------|----------|
| 0 | `black marble` | ⚫ Negra | 492 |
| 1 | `blue marble` | 🔵 Azul | 510 |
| 2 | `green marble` | 🟢 Verde | 485 |
| 3 | `white marble` | ⚪ Blanca | 518 |

**Total: 855 imágenes etiquetadas**

### Metricas del modelo
- mAP50-95: 0.95
- mAP50:    0.99
- Precisión: 0.97
- Recall:    0.98

- Precisión (Precision): Mide qué porcentaje de las detecciones realizadas por el modelo fueron correctas. Una alta precisión indica pocas falsas alarmas.

- Recall: Mide la capacidad del modelo para detectar todos los objetos reales presentes en las imágenes. Un alto recall indica pocas omisiones.

- mAP50: Evalúa la precisión promedio del modelo utilizando un umbral de IoU de 0.50. Indica qué tan bien detecta los objetos.

- mAP50-95: Evalúa el rendimiento del modelo utilizando múltiples niveles de IoU entre 0.50 y 0.95. Es una métrica más estricta y refleja la precisión general del detector.

---

## ⚡ API de Inferencia — `api.py`

Backend construido con **FastAPI** que expone el modelo YOLOv8s como servicio REST.

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Verifica el estado de la API, conexión y modelo YOLO cargado |
| `GET` | `/stats/totales` | Retorna el total acumulado de canicas detectadas por clase |
| `GET` | `/stats/historial` | Retorna historial paginado de detecciones registradas |
| `GET` | `/classes` | Lista las clases detectables del modelo YOLO |
| `GET` | `/metrics` | Retorna métricas generales de inferencia y rendimiento |
| `POST` | `/predict` | Recibe imagen → retorna JSON con detecciones |
| `POST` | `/predict/image` | Recibe imagen → retorna PNG anotado |
| `POST` | `/predict/video` | Recibe video → retorna MP4 anotado |
| `POST` | `/predict/camera` | Procesa captura de cámara → retorna detecciones |

### Ejemplo de respuesta `/predict`
```json
{
  "detections": [
    {
      "class_id": 1,
      "class_name": "blue marble",
      "confidence": 0.92,
      "x1": 120, "y1": 85, "x2": 200, "y2": 165,
      "width": 80, "height": 80
    }
  ],
  "inference_ms": 37.4,
  "total_detections": 1
}
```

### Parámetros de inferencia
- `conf` — umbral de confianza (default: 0.60)
- `iou` — umbral IoU para NMS (default: 0.45)

---

## 🖥️ Aplicación Web — `app.py`

Frontend construido con **Streamlit** en modo desacoplado — consume la API REST en vez de cargar el modelo localmente.

## 🖥️ Aplicación Movil — `ExpoGo`

App movil construida con ExpoGo que accede a la API y a postgres [EXPOGO](https://github.com/jjuandiego099/App-Movil-ExpoGo-Deteccion-de-Maras-o-Canicas).

### Tabs disponibles

#### 📷 Imagen
Sube una imagen en JPG, PNG, BMP o WEBP. La app muestra la imagen original lado a lado con la imagen anotada, métricas de detección (total, confianza media, tiempo de inferencia, resolución) y tabla detallada de cada detección con sus coordenadas.

#### 🎬 Video
Sube un video MP4, AVI, MOV o MKV. La API procesa frame a frame y retorna el video completo con las detecciones dibujadas. Incluye botón de descarga del video anotado.

#### 📹 Cámara en vivo
Dos modos:
- **Stream en tiempo real** usando `streamlit-webrtc` — cada frame se envía a la API y se devuelve anotado en vivo
- **Foto única** con `st.camera_input` — captura un momento y lo envía a la API

#### 📊 Estadísticas
Panel de resultados históricos conectado a PostgreSQL:
- Gráfico de barras comparativo entre las 4 clases con Plotly
- Resumen de totales y porcentajes por clase
- Tabla paginada (10 en 10) con historial completo de detecciones incluyendo fecha, fuente, conteo por clase, confianza promedio y tiempo de inferencia

### Configuración (sidebar)
- **Umbral de confianza** — slider 0.1 a 1.0 (recomendado 0.50–0.65)
- **Umbral IoU (NMS)** — slider 0.1 a 1.0 (recomendado 0.40–0.50)
- **Indicador de estado** de la API en tiempo real

---

## 🗄️ Base de Datos — PostgreSQL

Cada detección realizada desde cualquier tab (Imagen, Video, Cámara) se guarda automáticamente en PostgreSQL.

### Tabla `detecciones`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | SERIAL | ID autoincremental |
| `fecha` | TIMESTAMP | Fecha y hora de la detección |
| `fuente` | VARCHAR | `imagen`, `video` o `camara` |
| `verde` | INTEGER | Cantidad de maras verdes detectadas |
| `azul` | INTEGER | Cantidad de maras azules detectadas |
| `blanca` | INTEGER | Cantidad de maras blancas detectadas |
| `negra` | INTEGER | Cantidad de maras negras detectadas |
| `total` | INTEGER | Total de detecciones |
| `confianza_avg` | FLOAT | Confianza promedio |
| `inferencia_ms` | FLOAT | Tiempo de inferencia en ms |

---

## 🚀 Instalación local

### Requisitos
- Docker y Docker Compose
- Puerto 8000 (API), 8501 (Streamlit), 5432 (PostgreSQL)

### Levantar el proyecto

```bash
git clone https://github.com/tu-usuario/Modelo-de-Prediccion-de-Maras-Canicas.git
cd Modelo-de-Prediccion-de-Maras-Canicas
docker compose up -d --build
```

Accede a:
- **App:** http://localhost:8501
- **API docs:** http://localhost:8000/docs


### Crear la tabla en PostgreSQL

```bash
docker exec -it <nombre-contenedor-postgres> psql -U admin -d detecciones -c "
CREATE TABLE IF NOT EXISTS detecciones (
    id SERIAL PRIMARY KEY,
    fecha TIMESTAMP DEFAULT NOW(),
    fuente VARCHAR(20),
    verde INTEGER DEFAULT 0,
    azul INTEGER DEFAULT 0,
    blanca INTEGER DEFAULT 0,
    negra INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    confianza_avg FLOAT DEFAULT 0,
    inferencia_ms FLOAT DEFAULT 0
);"
```

---

## 📄 Licencia

Copyright (c) 2026
Universidad Autónoma de Bucaramanga (UNAB)
