# API Detección de Maras — Documentación

> **Versión:** 1.2.0 · **Modelo:** YOLOv8s · **Framework:** FastAPI  
> **Autores:** Juan Diego Chaparro García, Juan José Vargas, Santiago Amado

---

## Descripción general

API REST para la detección y clasificación de canicas (maras) en imágenes y videos usando el modelo YOLOv8s. Soporta inferencia sobre imágenes estáticas, videos completos y envío de resultados desde la app móvil. Los resultados se persisten automáticamente en una base de datos PostgreSQL.

**Clases detectadas:**

| ID | Nombre          | Color (BGR)       |
|----|-----------------|-------------------|
| 0  | Green Marble    | `(0, 229, 255)`   |
| 1  | Blue Marble     | `(255, 61, 87)`   |
| 2  | White Marble    | `(255, 214, 10)`  |
| 3  | Black Marble    | `(160, 110, 255)` |

---

## Variables de entorno

| Variable   | Valor por defecto | Descripción                      |
|------------|-------------------|----------------------------------|
| `DB_HOST`  | `postgres`        | Host del servicio PostgreSQL     |
| `DB_PORT`  | `5432`            | Puerto de PostgreSQL             |
| `DB_NAME`  | `detecciones`     | Nombre de la base de datos       |
| `DB_USER`  | `admin`           | Usuario de PostgreSQL            |
| `DB_PASS`  | `canicas123`      | Contraseña de PostgreSQL         |

---

## Endpoints

### Info

---

#### `GET /`

Retorna información general sobre la API, versión, modelo y lista de endpoints disponibles.

**Respuesta `200 OK`:**
```json
{
  "name": "API Detección de Maras",
  "version": "1.2.0",
  "model": "YOLOv8s",
  "classes": { "0": "green marble", "1": "blue marble", ... },
  "endpoints": { ... }
}
```

---

#### `GET /health`

Verifica el estado del servidor y confirma que el modelo está cargado.

**Respuesta `200 OK`:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": { "0": "green marble", ... },
  "num_classes": 4
}
```

**Respuesta `503 Service Unavailable`** si el modelo no está cargado.

---

### Inferencia

---

#### `POST /predict`

Realiza inferencia sobre una imagen y guarda el resultado en PostgreSQL. Retorna un JSON con todas las detecciones.

**Parámetros (form-data):**

| Parámetro | Tipo   | Requerido | Default    | Descripción                                      |
|-----------|--------|-----------|------------|--------------------------------------------------|
| `file`    | file   | ✅        | —          | Imagen JPG, PNG o BMP                            |
| `conf`    | float  | ❌        | `0.5`      | Umbral de confianza (0–1)                        |
| `iou`     | float  | ❌        | `0.45`     | Umbral de IoU para NMS (0–1)                     |
| `fuente`  | string | ❌        | `"imagen"` | Origen de la petición: `imagen`, `camara`, `movil` |

**Respuesta `200 OK`:**
```json
{
  "success": true,
  "inference_ms": 45.32,
  "total_detections": 3,
  "image_width": 640,
  "image_height": 480,
  "detections": [
    {
      "class_id": 0,
      "class_name": "green marble",
      "confidence": 0.9123,
      "x1": 120, "y1": 80, "x2": 200, "y2": 160,
      "width": 80, "height": 80
    }
  ]
}
```

**Notas:**
- El parámetro `fuente=movil` permite identificar las peticiones de la app móvil.
- El guardado en BD es automático; no se debe llamar también a `/save`.

---

#### `POST /predict/image`

Realiza inferencia sobre una imagen y retorna el **PNG anotado con bounding boxes**. No guarda en base de datos (el guardado lo maneja `/predict`).

**Parámetros (form-data):**

| Parámetro | Tipo  | Requerido | Default | Descripción           |
|-----------|-------|-----------|---------|-----------------------|
| `file`    | file  | ✅        | —       | Imagen JPG, PNG o BMP |
| `conf`    | float | ❌        | `0.5`   | Umbral de confianza   |
| `iou`     | float | ❌        | `0.45`  | Umbral de IoU         |

**Respuesta `200 OK`:**
- `Content-Type: image/png`
- Header `X-Detections`: número de detecciones encontradas

**Notas:**
- Pensado para ser llamado en paralelo desde el frontend solo para obtener la imagen anotada de calidad.

---

#### `POST /predict/video`

Realiza inferencia sobre un video completo, frame a frame. Retorna el **MP4 anotado**. Guarda un resumen acumulado en PostgreSQL al finalizar (`fuente='video'`).

**Parámetros (form-data):**

| Parámetro     | Tipo  | Requerido | Default | Descripción                                          |
|---------------|-------|-----------|---------|------------------------------------------------------|
| `file`        | file  | ✅        | —       | Video MP4, AVI, MOV o MKV                            |
| `conf`        | float | ❌        | `0.5`   | Umbral de confianza                                  |
| `iou`         | float | ❌        | `0.45`  | Umbral de IoU                                        |
| `skip_frames` | int   | ❌        | `1`     | Procesa 1 de cada N frames (1 = todos los frames)    |

**Respuesta `200 OK`:**
- `Content-Type: video/mp4`
- Headers informativos:

| Header               | Descripción                          |
|----------------------|--------------------------------------|
| `X-Total-Frames`     | Total de frames del video original   |
| `X-Processed-Frames` | Frames efectivamente procesados      |
| `X-Total-Detections` | Detecciones acumuladas en todo el video |
| `X-Processing-Ms`    | Tiempo total de procesamiento en ms  |

**Notas:**
- El video de salida es re-codificado con `libx264` + `yuv420p` vía ffmpeg para garantizar compatibilidad.
- El archivo de entrada soporta los MIME types: `video/mp4`, `video/avi`, `video/quicktime`, `video/x-matroska`, `video/x-msvideo`.

---

### Base de datos

---

#### `POST /save`

Endpoint exclusivo para la **app móvil (Expo)**. Recibe los conteos ya calculados en el cliente y los persiste directamente en PostgreSQL con `fuente='movil'`.

> ⚠️ No llamar a `/save` si ya se usó `POST /predict` con `fuente=movil`, pues generaría un registro duplicado.

**Cuerpo (JSON):**
```json
{
  "fuente":        "movil",
  "verde":         3,
  "azul":          1,
  "blanca":        0,
  "negra":         2,
  "total":         6,
  "confianza_avg": 0.8741,
  "inferencia_ms": 38.5
}
```

| Campo           | Tipo   | Default   | Descripción                          |
|-----------------|--------|-----------|--------------------------------------|
| `fuente`        | string | `"movil"` | Origen del registro                  |
| `verde`         | int    | `0`       | Cantidad de canicas verdes           |
| `azul`          | int    | `0`       | Cantidad de canicas azules           |
| `blanca`        | int    | `0`       | Cantidad de canicas blancas          |
| `negra`         | int    | `0`       | Cantidad de canicas negras           |
| `total`         | int    | `0`       | Total de detecciones                 |
| `confianza_avg` | float  | `0.0`     | Confianza promedio de las detecciones|
| `inferencia_ms` | float  | `0.0`     | Tiempo de inferencia en ms           |

**Respuesta `200 OK`:**
```json
{
  "success": true,
  "message": "Detección guardada correctamente"
}
```

**Respuesta `503`** si la BD no está disponible. **Respuesta `500`** si ocurre un error al insertar.

---

### Stats

---

#### `GET /stats/totales`

Retorna la suma acumulada de todas las detecciones registradas en la base de datos, agrupadas por tipo de canica.

**Respuesta `200 OK`:**
```json
{
  "verde":  142,
  "azul":   87,
  "blanca": 53,
  "negra":  210
}
```

---

#### `GET /stats/historial`

Retorna el historial paginado de detecciones, ordenado de más reciente a más antiguo.

**Parámetros (query string):**

| Parámetro | Tipo | Default | Descripción                        |
|-----------|------|---------|------------------------------------|
| `limit`   | int  | `10`    | Número máximo de registros a traer |
| `offset`  | int  | `0`     | Desplazamiento para paginación     |

**Respuesta `200 OK`:**
```json
{
  "total": 320,
  "rows": [
    {
      "id":     45,
      "fecha":  "05/22 14:30",
      "fuente": "imagen",
      "verde":  2,
      "azul":   0,
      "blanca": 1,
      "negra":  3,
      "total":  6
    }
  ]
}
```

---

## Esquema de la tabla `detecciones` (PostgreSQL)

| Columna         | Tipo      | Descripción                              |
|-----------------|-----------|------------------------------------------|
| `id`            | serial PK | Identificador autoincremental            |
| `fecha`         | timestamp | Fecha/hora del registro (zona America/Bogota) |
| `fuente`        | text      | `imagen`, `video`, `camara` o `movil`    |
| `verde`         | int       | Cantidad de canicas verdes               |
| `azul`          | int       | Cantidad de canicas azules               |
| `blanca`        | int       | Cantidad de canicas blancas              |
| `negra`         | int       | Cantidad de canicas negras               |
| `total`         | int       | Total de detecciones                     |
| `confianza_avg` | float     | Confianza promedio                       |
| `inferencia_ms` | float     | Tiempo de inferencia en milisegundos     |

---

## Documentación interactiva

| URL       | Descripción              |
|-----------|--------------------------|
| `/docs`   | Swagger UI (OpenAPI)     |
| `/redoc`  | ReDoc (alternativa)      |

---

## Códigos de error comunes

| Código | Causa                                         |
|--------|-----------------------------------------------|
| `400`  | Archivo inválido o no decodificable           |
| `503`  | Modelo YOLOv8s no cargado / BD no disponible  |
| `500`  | Error interno al guardar en PostgreSQL        |
