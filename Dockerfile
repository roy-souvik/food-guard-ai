FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for SQLite
RUN mkdir -p /data

# Expose Gradio port
EXPOSE 7860

# Set environment variables
ENV FOODGUARD_DB_PATH=/data/foodguard.db
ENV PYTHONUNBUFFERED=1

# Run Gradio app
CMD ["python", "app.py"]
