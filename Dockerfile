FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and to run in unbuffered mode
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# --- CORRECTED INSTALLATION ---
# Set environment variables needed to build llama-cpp-python for CPU
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=OFF"
ENV FORCE_CMAKE=1

# Install all Python dependencies from the requirements file in a single step
RUN pip install --no-cache-dir -r requirements.txt
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
RUN pip install llama-cpp-python
# Copy ONLY your application code
COPY app.py .

# Expose the port that Gradio will run on
EXPOSE 7860

# The command to run your Gradio app when the container starts
# The app will download the large files from the Hub on its own
CMD ["python", "app.py"]
