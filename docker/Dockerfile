FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY ./app ./app
COPY ./notebooks ./notebooks
COPY ./scripts ./scripts
COPY ./tests ./tests
COPY ./.env.example ./.env.example

EXPOSE 8501 8888

# Crear directorio para logs
RUN mkdir -p /var/log

# Crear directorio para configuración de Jupyter
RUN mkdir -p /root/.jupyter

# Crear configuración de Jupyter para permitir acceso sin token
RUN echo "c.ServerApp.token = ''" > /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.notebook_dir = '/app'" >> /root/.jupyter/jupyter_lab_config.py

# Copiar y configurar el script de entrada
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
