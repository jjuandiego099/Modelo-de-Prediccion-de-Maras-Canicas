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
);