# 🚀 Despliegue en AWS EC2 — Guía completa

> Documentación del proceso completo de despliegue del sistema de Detección de Maras en AWS EC2 con Docker, SSL/HTTPS mediante DuckDNS + Let's Encrypt, y todos los servicios dockerizados (API, Streamlit, PostgreSQL, n8n).

**URL de producción:** [https://deteccion-maras-canicas.duckdns.org](https://deteccion-maras-canicas.duckdns.org)

---

## 🏗️ Arquitectura general

```
Usuario (navegador)
        │
        │ HTTPS :443
        ▼
┌─────────────────────────────────────────┐
│              AWS EC2 (Ubuntu 24)        │
│                                         │
│   Nginx (fuera de Docker)               │
│   └── SSL via Let's Encrypt             │
│         │                               │
│         ├──► Docker: Streamlit :8501    │
│         ├──► Docker: API FastAPI :8000  │     
│         └──► Docker: PostgreSQL :5432   │
│                                         │
│   Todos los puertos internos en         │
│   127.0.0.1 — no expuestos al exterior  │
└─────────────────────────────────────────┘
```

---

## ☁️ 1. Instancia EC2 en AWS

### Especificaciones usadas
- **AMI:** Ubuntu Server 24.04 LTS
- **Tipo:** t2.large (o superior según carga)
- **Almacenamiento:** 50 GB SSD
- **IP pública:** `3.212.170.240`

> Se recomienda activar una **IP elastica** para la configuracion de EC2.

### Security Group — puertos abiertos
| Puerto | Protocolo | Descripción |
|--------|-----------|-------------|
| 22 | TCP | SSH para administración |
| 80 | TCP | HTTP (necesario para validación de Certbot) |
| 443 | TCP | HTTPS (tráfico de producción) |

> ⚠️ Los puertos 8000, 8501, 5432 y 5678 **NO están abiertos** al exterior — solo son accesibles internamente por Nginx.

---

## 🐳 2. Docker y Docker Compose

Todos los servicios corren como contenedores Docker orquestados con Docker Compose.

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
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: canicas123
      POSTGRES_DB: detecciones
    ports:
      # Restringido a localhost igual que los demás servicios
      - "127.0.0.1:5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    # Healthcheck para saber cuándo Postgres está realmente listo para aceptar conexiones
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d detecciones"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    container_name: maras-api
    command: uvicorn api:app --host 0.0.0.0 --port 8000
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - .:/app
    environment:
      - PYTHONUNBUFFERED=1
    depends_on:
      postgres:
        condition: service_healthy   # Espera a que Postgres esté listo
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  streamlit:
    build: .
    container_name: maras-app
    command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
    ports:
      - "127.0.0.1:8501:8501"
    volumes:
      - .:/app
    environment:
      - PYTHONUNBUFFERED=1
      # Variables de BD que app.py lee con os.getenv()
      - DB_PORT=5432
      - DB_NAME=detecciones
      - DB_USER=admin
      - DB_PASS=canicas123
    depends_on:
      api:
        condition: service_healthy   # Espera a que la API haya cargado el modelo
      postgres:
        condition: service_healthy   # Espera a que Postgres esté listo
    restart: unless-stopped

volumes:
  postgres_data:

### Levantar todos los servicios

```bash
cd Modelo-de-Prediccion-de-Maras-Canicas
sudo docker compose up -d --build
```

### Comandos útiles

```bash
sudo docker compose ps          # ver estado de todos los contenedores
sudo docker compose logs -f     # ver logs en tiempo real
sudo docker compose restart streamlit   # reiniciar solo Streamlit
sudo docker compose down        # detener todo
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

Para habilitar HTTPS (requerido por Chrome para acceder a cámara y micrófono), se instaló **Nginx** fuera de Docker como proxy inverso, y **Certbot** para gestionar el certificado SSL gratuito de Let's Encrypt.

### ¿Por qué HTTPS es obligatorio?
Los navegadores modernos (Chrome, Firefox, Safari) bloquean el acceso a cámara y micrófono en sitios HTTP. Solo permiten estos permisos en sitios con HTTPS o en `localhost`.

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
    listen 80;
    server_name deteccion-maras-canicas.duckdns.org;

    # Streamlit — app principal
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # API FastAPI
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # n8n — automatización
    location /n8n/ {
        proxy_pass http://localhost:5678/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
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
sudo docker exec -it modelo-de-prediccion-de-maras-canicas-postgres-1 \
  psql -U admin -d detecciones
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
Streamlit se conecta a PostgreSQL usando el **nombre del servicio** `postgres` como host — esto funciona gracias a la red interna de Docker Compose donde los contenedores se resuelven por nombre:

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


## 🔁 6. Flujo de actualización (CI/CD manual)

Para actualizar la aplicación en producción:

```bash
# 1. En local — hacer cambios y push
git add .
git commit -m "descripcion del cambio"
git push origin main

# 2. En EC2 — pull y rebuild
cd Modelo-de-Prediccion-de-Maras-Canicas
git pull origin main
sudo docker compose up -d --build streamlit   # rebuild solo del servicio modificado
```

---

## ✅ URLs de producción

| Servicio | URL |
|----------|-----|
| App Streamlit | https://deteccion-maras-canicas.duckdns.org |
| API docs | http://localhost:8000/docs (solo desde EC2) |

---

## 🛡️ Seguridad implementada

- Todos los puertos internos (8000, 8501, 5432, 5678) vinculados a `127.0.0.1` — inaccesibles desde internet
- Un único punto de entrada público: Nginx en puerto 443 con HTTPS
- Certificado SSL gratuito con renovación automática cada 90 días
- Security Group de EC2 con solo puertos 22, 80 y 443 abiertos
