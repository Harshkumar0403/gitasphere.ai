FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- CORRECTED INSTALLATION ---

# Step 1: Install system dependencies needed to COMPILE packages if a pre-built wheel isn't found.
# This makes the build process more reliable.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Step 2: Install all Python packages in a single command.
# This will TRY to use the fast pre-built wheels first, but can compile if needed.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY app.py .

EXPOSE 8080

CMD ["python", "app.py"]
