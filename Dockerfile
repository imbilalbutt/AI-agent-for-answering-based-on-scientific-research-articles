# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /src/docs ./chroma_db

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV RELOAD=false

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "fastapi_rag.py"]