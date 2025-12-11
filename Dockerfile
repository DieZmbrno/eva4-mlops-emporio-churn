# Imagen base de Python (ligera)
FROM python:3.11-slim

# Evitar que Python genere archivos .pyc y buffering raro
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos solo requirements primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalamos dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código de la app
COPY src ./src

# Copiamos el modelo entrenado
COPY models ./models

# (Opcional) Copiar reportes si quieres tenerlos dentro del contenedor
# COPY reports ./reports

# Exponemos el puerto donde correrá Uvicorn
EXPOSE 8000

# Comando por defecto: lanzar la API con Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
