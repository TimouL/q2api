FROM python:3.11-slim

WORKDIR /app

# Install tools required at runtime
RUN apt-get update && apt-get install -y curl git ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .
COPY templates ./templates
COPY frontend ./frontend

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]