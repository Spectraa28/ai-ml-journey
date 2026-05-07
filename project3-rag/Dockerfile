# 1. Use a slim Python 3.11 base for a lightweight image
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files and buffering logs
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies (needed for certain ML library wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
# We copy this separately to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import chromadb; client = chromadb.Client(); client.heartbeat()"

# 6. ML PRE-CACHE: Bake the model weights into the image layer.
# This ensures the 400MB model is ready at launch, zero download delay.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()"

# 7. Copy the application source code
COPY . .

# 8. Expose the FastAPI port
EXPOSE 8000

# 9. Launch the service
# We use the "module:app" string syntax for better worker management
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

ENV GIT_PYTHON_REFRESH=quiet