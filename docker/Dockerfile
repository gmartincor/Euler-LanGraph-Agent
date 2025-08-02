FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    netcat-traditional \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

# Create Jupyter configuration directory and copy config
RUN mkdir -p /root/.jupyter
COPY ./docker/jupyter_lab_config.py /root/.jupyter/jupyter_lab_config.py

# Copy application files
COPY ./app ./app
COPY ./notebooks ./notebooks
COPY ./scripts ./scripts
COPY ./tests ./tests
COPY ./.env.example ./.env.example

EXPOSE 8501 8888

# Create directories for logs and workspace
RUN mkdir -p /var/log /app/.jupyter/workspaces

# Crear directorio para configuración de Jupyter
RUN mkdir -p /root/.jupyter

# Crear configuración de Jupyter optimizada para VS Code
RUN echo "c.ServerApp.token = ''" > /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.notebook_dir = '/app'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_origin = '*'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.disable_check_xsrf = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.KernelManager.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py

# Copiar y configurar el script de entrada
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
