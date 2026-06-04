# Use an optimized, official Python slim runtime environment core
FROM python:3.12-slim

# Set strict system configurations to maximize logging streams
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish the interior container workspace filesystem tracks
WORKDIR /app

# Install native Linux database drivers and system communication requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /lib/apt/lists/*

# Copy the dependency tracking configuration manifest sheet over first
COPY requirements.txt .

# Execute high-speed cached dependency pip compilation layers
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the operational software application trees into the image
COPY . .

# Expose the internal network binding communication routing port interface
EXPOSE 8000

# Fire up the production enterprise web server worker interface on boot
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
