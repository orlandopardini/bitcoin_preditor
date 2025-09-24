# Dockerfile — imagem base para todos os serviços
FROM python:3.11-slim

# set UTF-8
ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only requirements first (cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the repository
COPY . /app

# Create DB and logs directories (mounted volumes can override)
RUN mkdir -p /app/GIT/DB /app/logs /app/.pids

# Default command — no-op. Each service will override
CMD ["bash", "-c", "echo 'Image ready' && sleep infinity"]
