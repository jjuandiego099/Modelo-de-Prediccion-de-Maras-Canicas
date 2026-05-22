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
        │   Nginx (fuera de Docker)               │
        │   └── SSL via Let's Encrypt             │
        │         │                               │
        │         ├──► /        → Streamlit :8501 │
        │         ├──► /api/    → FastAPI  :8000  │
        │                                         │
        │                                         │
        │   Docker (red interna)                  │
        │   ├── maras-api      (FastAPI + YOLO)   │
        │   ├── maras-app      (Streamlit)        │
        │   ├── maras-db       (PostgreSQL)       │
        │                                         │
        │                                         │
        │   Todos los puertos en 127.0.0.1        │
        │   — no expuestos al exterior            │
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

Todos los servicios corren como contenedores Docker orquestados con Docker Compose. Los puertos están vinculados a `127.0.0.1` para que solo Nginx (instalado en el sistema) pueda acceder a ellos.

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

    # Persistencia de datos de la base de datos
    volumes:
      - postgres_data:/var/lib/postgresql/data

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

## 🔒 4. HTTPS con Nginx + Let's Encrypt (Certbot)

Nginx se instaló **directamente en el sistema EC2** (fuera de Docker) como proxy inverso único. Certbot gestiona el certificado SSL gratuito de Let's Encrypt.

### ¿Por qué HTTPS es obligatorio?
- Los navegadores modernos bloquean acceso a cámara y micrófono en sitios HTTP.
- **Expo Go** en iOS y Android bloquea requests HTTP por defecto — solo acepta HTTPS.

### Instalación de Nginx y Certbot

```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx -y
```

### Configuración de Nginx

```bash
sudo nano /etc/nginx/sites-available/marbles
```

```nginx
server {
    server_name deteccion-maras-canicas.duckdns.org;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /n8n/ {
        proxy_pass http://localhost:5678/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/deteccion-maras-canicas.duckdns.org/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/deteccion-maras-canicas.duckdns.org/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot

}
server {
    if ($host = deteccion-maras-canicas.duckdns.org) {
        return 301 https://$host$request_uri;
    } # managed by Certbot
```

```bash
# Activar configuración
sudo ln -s /etc/nginx/sites-available/marbles /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default   # desactivar página por defecto
sudo nginx -t                              # verificar sintaxis
sudo systemctl reload nginx
```

### Obtener certificado SSL gratuito

```bash
sudo certbot --nginx -d deteccion-maras-canicas.duckdns.org
```

Certbot automáticamente:
- Verifica que el dominio apunta a la EC2
- Genera el certificado SSL de Let's Encrypt (gratuito, válido 90 días)
- Modifica la config de Nginx para escuchar en el puerto 443
- Configura redirección automática HTTP → HTTPS
- Programa renovación automática del certificado

### Verificar renovación automática

```bash
sudo certbot renew --dry-run
```

---

## 🗄️ 5. PostgreSQL dockerizado

La base de datos corre como contenedor Docker con volumen persistente, lo que significa que los datos sobreviven reinicios del contenedor.

### Crear la tabla de detecciones

```bash
sudo docker exec -it maras-db psql -U admin -d detecciones
```

```sql
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
3. Lo redirige internamente a `localhost:8000` (FastAPI en Docker)
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

- Todos los puertos internos (8000, 8501, 5432, 5678) vinculados a `127.0.0.1` — inaccesibles desde internet
- Un único punto de entrada público: Nginx en puerto 443 con HTTPS
- Certificado SSL gratuito con renovación automática cada 90 días
- Security Group de EC2 con solo puertos 22, 80 y 443 abiertos
- Expo Go usa HTTPS obligatoriamente — no hay tráfico HTTP expuesto