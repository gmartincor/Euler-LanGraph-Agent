FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY ./app ./app
COPY ./notebooks ./notebooks
COPY ./scripts ./scripts

EXPOSE 8501 8888

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

# Crear script de inicio que ejecute tanto Streamlit como Jupyter
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'set -e' >> /app/start.sh && \
    echo 'echo "🚀 Starting Jupyter Lab on port 8888..."' >> /app/start.sh && \
    echo 'poetry run jupyter lab --config=/root/.jupyter/jupyter_lab_config.py &' >> /app/start.sh && \
    echo 'echo "🚀 Starting Streamlit on port 8501..."' >> /app/start.sh && \
    echo 'poetry run streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0 &' >> /app/start.sh && \
    echo 'echo "✅ Services started. Access:"' >> /app/start.sh && \
    echo 'echo "  📊 Streamlit: http://localhost:8501"' >> /app/start.sh && \
    echo 'echo "  📝 Jupyter Lab: http://localhost:8888"' >> /app/start.sh && \
    echo 'wait' >> /app/start.sh && \
    chmod +x /app/start.sh

CMD ["/app/start.sh"]
