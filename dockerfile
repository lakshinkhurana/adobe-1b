# Use official Python slim base image
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to cache install layer
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time to avoid internet requirement at runtime
RUN python -m nltk.downloader punkt -d /root/nltk_data

# Copy all project files
COPY . .

# Set environment for NLTK data
ENV NLTK_DATA=/root/nltk_data

# Run script with CMD (you can override this at runtime)
CMD ["python", "your_script_name.py"]
