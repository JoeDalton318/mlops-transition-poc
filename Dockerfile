FROM python:3.10-slim

# Use a specific non-root user for security.
RUN groupadd --system mlopsuser && useradd --system --gid mlopsuser --create-home --home-dir /home/mlopsuser mlopsuser

WORKDIR /app

# Install only required system dependencies and clean package cache.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src

RUN chown -R mlopsuser:mlopsuser /app

USER mlopsuser

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
