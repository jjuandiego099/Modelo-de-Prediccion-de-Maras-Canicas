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
├── Dockerfile              # Imagen Docker unificada (API + Streamlit + Nginx)
├── docker-compose.yml      # Orquestación de todos los servicios
├── .dockerignore           # Archivos ignorados por Docker
├── nginx/                  # Configuración de Nginx (incluida en el repo)
│   └── nginx.conf          # Proxy inverso, SSL y límites de subida
├── init.sql                # Script SQL — crea la tabla automáticamente al iniciar
├── MD/                     # Documentación adicional
│   ├── API.md              # Referencia completa de la API
│   └── DEPLOYMENT.md       # Guía de despliegue en AWS EC2
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

### 🧠 Dataset de Maras o Canicas 
[Dataset](https://app.roboflow.com/juan-jose-vargas-correa-s-workspace/deteccion-maras-yb0z3/4)

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
| `GET` | `/` | Info general de la API y lista de endpoints |
| `GET` | `/health` | Verifica el estado de la API y modelo YOLO cargado |
| `GET` | `/stats/totales` | Retorna el total acumulado de canicas detectadas por clase |
| `GET` | `/stats/historial` | Retorna historial paginado de detecciones registradas |
| `POST` | `/predict` | Recibe imagen → retorna JSON con detecciones (guarda en BD) |
| `POST` | `/predict/image` | Recibe imagen → retorna PNG anotado (no guarda en BD) |
| `POST` | `/predict/video` | Recibe video → retorna MP4 anotado (no guarda en BD) |
| `POST` | `/save` | Guarda conteos enviados por la app móvil directamente en BD |

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


#### 📊 Estadísticas
Panel de resultados históricos conectado a PostgreSQL:
- Gráfico de barras comparativo entre las 4 clases con Plotly
- Resumen de totales y porcentajes por clase
- Tabla paginada (10 en 10) con historial completo de detecciones incluyendo fecha, fuente, conteo por clase, confianza promedio y tiempo de inferencia
  
  
#### ℹ️ Info
Describe brevemente el proyecto, las categorias de canicas a detectar y una breve explcacion de las variables de configuración 
- Descripción del proyecto
- Tecnologias utilizadas
- Categorias de maras o canicas
- Explicación de las varables de configuración(Umbral de confianz y Umbral de IoU)

  
### Configuración (sidebar)
- **Umbral de confianza** — slider 0.1 a 1.0 (recomendado 0.50–0.65)
- **Umbral IoU (NMS)** — slider 0.1 a 1.0 (recomendado 0.40–0.50)
- **Indicador de estado** de la API en tiempo real

---

## 🗄️ Base de Datos — PostgreSQL

Las detecciones de **Imagen** y **Cámara** se guardan automáticamente en PostgreSQL. Las detecciones de **Video** no se persisten en BD.

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


### Tabla en PostgreSQL

La tabla `detecciones` se crea automáticamente al levantar los contenedores gracias al archivo `init.sql` montado en el servicio `postgres`. No es necesario crearla manualmente.

---

## 📄 Licencia

Copyright (c) 2026
Universidad Autónoma de Bucaramanga (UNAB)
