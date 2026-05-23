# 🚀 Despliegue en AWS EC2 — Guía completa

> Documentación del proceso completo de despliegue del sistema de Detección de Maras en AWS EC2 con Docker, SSL/HTTPS mediante DuckDNS + Let's Encrypt, todos los servicios dockerizados (API, Streamlit, PostgreSQL) y la app móvil con Expo Go.

**URL de producción:** [https://deteccion-maras-canicas.duckdns.org](https://deteccion-maras-canicas.duckdns.org)

---

## 🏗️ Arquitectura general

```
📱 Expo Go (celular)          🌐 Navegador (PC)
        │                             │
        │ HTTPS :443                  │ HTTPS :443
        └──────────────┬──────────────┘
                       ▼
        ┌─────────────────────────────────────────┐
        │              AWS EC2 (Ubuntu 24)        │
        │                                         │
        │   Docker (red interna "interna")        │
        │   ├── maras-nginx    (Nginx + SSL)      │
        │   │   └── SSL via Let's Encrypt         │
        │   │         │                           │
        │   │         ├──► /      → maras-app     │
        │   │         ├──► /api/  → maras-api     │
        │   │                                     │
        │   ├── maras-api      (FastAPI + YOLO)   │
        │   ├── maras-app      (Streamlit)        │
        │   └── maras-postgres (PostgreSQL)       │
        │                                         │
        │   Puertos 80 y 443 expuestos al exterior│
        │   El resto solo en red interna Docker   │
        └─────────────────────────────────────────┘
```

---

## ☁️ 1. Instancia EC2 en AWS

### Especificaciones usadas
- **AMI:** Ubuntu Server 24.04 LTS
- **Tipo:** t2.large (o superior según carga)
- **Almacenamiento:** 50 GB SSD
- **IP pública:** `3.212.170.240` (IP elástica)

> Se recomienda activar una **IP elástica** para que la IP no cambie al reiniciar la instancia.

### Security Group — puertos abiertos
| Puerto | Protocolo | Descripción |
|--------|-----------|-------------|
| 22 | TCP | SSH para administración |
| 80 | TCP | HTTP (necesario para validación de Certbot) |
| 443 | TCP | HTTPS (tráfico de producción) |

> ⚠️ Los puertos 8000, 8501, 5432 y 5678 **NO están abiertos** al exterior — solo son accesibles internamente por Nginx.

---

## 🐳 2. Docker y Docker Compose

Todos los servicios corren como contenedores Docker orquestados con Docker Compose, incluyendo Nginx. La red interna de Docker (`interna`) comunica los contenedores entre sí sin exponer puertos al exterior, excepto el 80 y 443 de Nginx.

### Instalación de Docker en EC2

```bash
sudo apt update
sudo apt install docker.io docker-compose-plugin -y
sudo systemctl enable docker
sudo systemctl start docker
```

### `docker-compose.yml`

```yaml
services:

  postgres:
    image: postgres:15
    container_name: maras-postgres

    # Variables de entorno de PostgreSQL
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: canicas123
      POSTGRES_DB: detecciones
      PGTZ: America/Bogota

    # Solo expone PostgreSQL localmente en la EC2
    # No es accesible desde internet
    ports:
      - "127.0.0.1:5432:5432"

    # Persistencia de datos y script de inicialización automática
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

    # Reinicia automáticamente si falla
    restart: unless-stopped

    # Red interna Docker
    networks:
      - interna

    # Verifica que PostgreSQL esté listo
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d detecciones"]
      interval: 10s
      timeout: 5s
      retries: 5


  api:
    build: .
    container_name: maras-api

    # Ejecuta FastAPI con Uvicorn
    command: uvicorn api:app --host 0.0.0.0 --port 8000

    # Sincroniza archivos locales con el contenedor
    volumes:
      - .:/app

    # Variables de entorno usadas por FastAPI
    environment:
      - PYTHONUNBUFFERED=1
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=detecciones
      - DB_USER=admin
      - DB_PASS=canicas123
      - TZ=America/Bogota

    # Espera a que PostgreSQL esté listo
    depends_on:
      postgres:
        condition: service_healthy

    restart: unless-stopped

    # Red interna Docker
    networks:
      - interna

    # Comprueba que FastAPI responda correctamente
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3


  streamlit:
    build: .
    container_name: maras-app

    # Ejecuta la interfaz Streamlit
    command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

    # Comparte archivos locales con el contenedor
    volumes:
      - .:/app

    # Variables de entorno usadas por Streamlit
    environment:
      - PYTHONUNBUFFERED=1
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=detecciones
      - DB_USER=admin
      - DB_PASS=canicas123
      - TZ=America/Bogota
      - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500

    # Espera a que API y PostgreSQL estén listos
    depends_on:
      api:
        condition: service_healthy
      postgres:
        condition: service_healthy

    restart: unless-stopped

    # Red interna Docker
    networks:
      - interna


  nginx:
    image: nginx:alpine
    container_name: maras-nginx

    # Expone HTTP y HTTPS al exterior
    ports:
      - "80:80"
      - "443:443"

    # Configuración de Nginx y certificados SSL
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro

    # Espera a que API y Streamlit estén listos
    depends_on:
      - api
      - streamlit

    restart: unless-stopped

    # Red interna Docker
    networks:
      - interna


# Red privada compartida entre contenedores
networks:
  interna:
    driver: bridge


# Volumen persistente para PostgreSQL
volumes:
  postgres_data:
```

### Levantar todos los servicios

```bash
cd Modelo-de-Prediccion-de-Maras-Canicas
sudo docker compose up -d --build
```

### Comandos útiles

```bash
sudo docker compose ps                        # ver estado de todos los contenedores
sudo docker compose logs -f                   # ver logs en tiempo real
sudo docker compose restart streamlit         # reiniciar solo Streamlit
sudo docker compose down                      # detener todo
```

---

## 🌐 3. Dominio gratuito con DuckDNS

Como no se contaba con un dominio propio, se usó **DuckDNS** — un servicio gratuito de DNS dinámico.

### Pasos realizados
1. Ingresar a [duckdns.org](https://www.duckdns.org) e iniciar sesión con Google
2. Crear el subdominio `deteccion-maras-canicas`
3. Asignar la IP pública de EC2: `3.212.170.240`
4. El dominio resultante: `deteccion-maras-canicas.duckdns.org`

> DuckDNS apunta el subdominio a la IP pública de la instancia EC2. Es gratuito y no requiere tarjeta de crédito.

---

## 🔒 4. HTTPS con Nginx dockerizado + Let's Encrypt (Certbot)

Nginx corre como **contenedor Docker** (`maras-nginx`) dentro de la misma red interna. Certbot se instala en el sistema EC2 solo para generar y renovar los certificados SSL, que luego se montan como volumen de solo lectura en el contenedor.

### ¿Por qué HTTPS es obligatorio?
- Los navegadores modernos bloquean acceso a cámara y micrófono en sitios HTTP.
- **Expo Go** en iOS y Android bloquea requests HTTP por defecto — solo acepta HTTPS.

### Obtener certificado SSL (solo la primera vez)

```bash
sudo apt update
sudo apt install certbot -y
# Detener cualquier proceso en el puerto 80 antes de correr certbot
sudo certbot certonly --standalone -d deteccion-maras-canicas.duckdns.org
```

Los certificados quedan en `/etc/letsencrypt/` y se montan en el contenedor Nginx como volumen de solo lectura (ver `docker-compose.yml`).

### Configuración de Nginx — `nginx/nginx.conf`

```nginx
events {
    worker_connections 1024;
}
http {
    client_max_body_size 500M;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    server {
        listen 80;
        server_name deteccion-maras-canicas.duckdns.org;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl;
        server_name deteccion-maras-canicas.duckdns.org;
        ssl_certificate     /etc/letsencrypt/live/deteccion-maras-canicas.duckdns.org/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/deteccion-maras-canicas.duckdns.org/privkey.pem;

        location / {
            proxy_pass http://maras-app:8501;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }

        location /api/ {
            proxy_pass http://maras-api:8000/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

    }
}
```

### Recargar Nginx sin bajar contenedores

```bash
docker compose exec nginx nginx -s reload
```

### Renovar certificado SSL

```bash
# Detener nginx para liberar el puerto 80
docker compose stop nginx
sudo certbot renew
docker compose start nginx
```

---

## 🗄️ 5. PostgreSQL dockerizado

La base de datos corre como contenedor Docker con volumen persistente, lo que significa que los datos sobreviven reinicios del contenedor.

### Inicialización automática de la tabla

La tabla `detecciones` se crea automáticamente la primera vez que arranca el contenedor. El archivo `init.sql` del repositorio se monta en `/docker-entrypoint-initdb.d/` — PostgreSQL ejecuta cualquier `.sql` en esa ruta al inicializarse por primera vez.

```sql
-- init.sql (incluido en el repositorio)
CREATE TABLE IF NOT EXISTS detecciones (
    id            SERIAL PRIMARY KEY,
    fecha         TIMESTAMP DEFAULT NOW(),
    fuente        VARCHAR(20),
    verde         INTEGER DEFAULT 0,
    azul          INTEGER DEFAULT 0,
    blanca        INTEGER DEFAULT 0,
    negra         INTEGER DEFAULT 0,
    total         INTEGER DEFAULT 0,
    confianza_avg FLOAT DEFAULT 0,
    inferencia_ms FLOAT DEFAULT 0
);
```

> ⚠️ El script solo se ejecuta si el volumen `postgres_data` está vacío (primera vez). Si el contenedor ya tiene datos, no lo vuelve a correr.

### Conexión desde Streamlit

Streamlit se conecta a PostgreSQL usando el **nombre del servicio** `postgres` como host — esto funciona gracias a la red interna de Docker Compose:

```python
psycopg2.connect(
    host="postgres",    # nombre del servicio en docker-compose
    port=5432,
    dbname="detecciones",
    user="admin",
    password="canicas123"
)
```

---

## 📱 6. App móvil con Expo Go

La app móvil se desarrolló con **React Native + Expo** y consume la misma API FastAPI a través de HTTPS.

### Requisitos
- Node.js instalado en la PC de desarrollo
- App **Expo Go** instalada en el celular (iOS o Android)
- PC y celular en la misma red, o usar túnel de Expo

### Instalación y arranque

```bash
# Crear proyecto (solo la primera vez)
npx create-expo-app App-Movil-Deteccon-Maras-ExpoGo --template (SDK 54)
cd App-Movil-Deteccon-Maras-ExpoGo

# Instalar dependencias
npx expo install expo-image-picker @react-native-community/slider

# Arrancar servidor de desarrollo
npx expo start --tunnel   # --tunnel permite usar datos móviles sin misma red WiFi
```

### Escanear QR
- **Android:** Abrir Expo Go → "Scan QR code"
- **iOS:** Abrir la cámara normal → apuntar al QR

### Estructura de la app

```
app/
├── (tabs)/
│   ├── _layout.tsx         ← define las pestañas (Detector, Estadisticas e Info)
│   ├── index.tsx           ← pantalla principal: detector con cámara/galería
│   ├── estadistcas.tsx     ← historial de las detecciones anteriores
│   └── info.tsx            ← información del proyecto y explicación de parámetros
```

### URL de la API en la app

```typescript
// app/(tabs)/index.tsx
const API_URL = "https://deteccion-maras-canicas.duckdns.org/api";
```

La app funciona porque:
1. Expo Go hace requests HTTPS a `deteccion-maras-canicas.duckdns.org`
2. Nginx recibe el request en el puerto 443
3. Lo redirige internamente al contenedor `maras-api:8000` (FastAPI)
4. FastAPI corre el modelo YOLOv8s y responde con las detecciones

### Parámetros configurables en la app
| Parámetro | Descripción | Valor recomendado |
|-----------|-------------|-------------------|
| Confianza | Certeza mínima para reportar una detección | 0.50 – 0.65 |
| IoU (NMS) | Umbral para eliminar cajas duplicadas | 0.40 – 0.50 |

---

## 🔁 7. Flujo de actualización (CI/CD manual)

Para actualizar la aplicación en producción:

```bash
# 1. En local — hacer cambios y push
git add .
git commit -m "descripcion del cambio"
git push origin main

# 2. En EC2 — pull y rebuild
cd Modelo-de-Prediccion-de-Maras-Canicas
git pull origin main
sudo docker compose up -d --build   # rebuild de los servicios modificados
```

---

## ✅ URLs de producción

| Servicio | URL |
|----------|-----|
| App Streamlit | https://deteccion-maras-canicas.duckdns.org |
| API FastAPI | https://deteccion-maras-canicas.duckdns.org/api/ |
| API Docs (Swagger) | https://deteccion-maras-canicas.duckdns.org/api/docs |
| App móvil (Expo Go) | Escanear QR al correr `npx expo start --tunnel` |
| App móvil (Repositorio) | [App Movil](https://github.com/jjuandiego099/App-Movil-ExpoGo-Deteccion-de-Maras-o-Canicas) |


---

## 🛡️ Seguridad implementada

- Todos los puertos internos (8000, 8501, 5432) accesibles solo dentro de la red interna Docker — no expuestos al exterior
- Un único punto de entrada público: Nginx en puerto 443 con HTTPS
- Certificado SSL gratuito con renovación automática cada 90 días
- Security Group de EC2 con solo puertos 22, 80 y 443 abiertos
- Expo Go usa HTTPS obligatoriamente — no hay tráfico HTTP expuesto